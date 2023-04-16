import numpy as np
import torch
from torch.utils.data import DataLoader
import util.jitfunc as f
from util import SPACING32
from util.taalhelper import augments_forward
from torch import nn
from sklearn.metrics import pairwise_distances


class LimitSortedList(object):

    def __init__(self, limit, descending=False) -> None:
        self.descending = descending
        self.limit = limit
        self._data = []

    def reset(self):
        self._data.clear()

    @property
    def data(self):
        return map(lambda x: int(x[0]), self._data)

    def extend(self, idx):
        assert isinstance(idx, (torch.Tensor, np.ndarray, list, tuple))
        idx = list(idx)
        self._data.extend(idx)
        if len(self._data) > self.limit:
            self._data = sorted(self._data, key=lambda x: x[1], reverse=self.descending)[:self.limit]


class QueryStrategy(object):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader) -> None:
        super().__init__()
        self.unlabeled_dataloader = unlabeled_dataloader
        self.labeled_dataloader = labeled_dataloader

    def select_dataset_idx(self, query_num):
        raise NotImplementedError

    def convert2img_idx(self, ds_idx, dataloader: DataLoader):
        return [dataloader.sampler.indices[img_id] for img_id in ds_idx]

    def sample(self, query_num):
        dataset_idx = self.select_dataset_idx(query_num)
        img_idx = self.convert2img_idx(dataset_idx, self.unlabeled_dataloader)
        self.labeled_dataloader.sampler.indices.extend(img_idx)
        # 注意这里不可以用index来移除，因为pop一个之后，原数组就变换了
        # for i in dataset_idx:
        #     self.unlabeled_dataloader.sampler.indices.pop(i)
        for item in img_idx:
            self.unlabeled_dataloader.sampler.indices.remove(item)


class RandomQuery(QueryStrategy):
    def sample(self, query_num):
        np.random.shuffle(self.unlabeled_dataloader.sampler.indices)
        self.labeled_dataloader.sampler.indices.extend(self.unlabeled_dataloader.sampler.indices[:query_num])
        del self.unlabeled_dataloader.sampler.indices[:query_num]


class SimpleQueryStrategy(QueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader)
        assert "trainer" in kwargs
        assert "descending" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.descending = kwargs["descending"]

    def compute_score(self, model_output):
        raise NotImplementedError

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        q = LimitSortedList(limit=query_num, descending=self.descending)
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(device)
            output = self.model(img)
            score = self.compute_score(output).cpu()
            assert score.shape[0] == img.shape[0], "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack([torch.arange(offset, offset + img.shape[0]), score]).data
            q.extend(idx_entropy)
        return q.data


class MaxEntropy(SimpleQueryStrategy):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=True, **kwargs)

    def compute_score(self, model_output):
        model_output = model_output.softmax(dim=1)
        return f.max_entropy(model_output, SPACING32)


class MarginConfidence(SimpleQueryStrategy):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        model_output = model_output.softmax(dim=1)
        return f.margin_confidence(model_output)


class LeastConfidence(SimpleQueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        model_output = model_output.softmax(dim=1)
        output_max = torch.max(model_output, dim=1)[0]
        return output_max.mean(dim=(-2, -1))


class TAAL(QueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader)
        assert "trainer" in kwargs
        self.model = kwargs["trainer"].model
        self.num_augmentations = int(kwargs.get("num_augmentations", 10))

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        q = LimitSortedList(limit=query_num, descending=True)
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(device)
            output = self.model(img).softmax(dim=1)
            aug_out = augments_forward(img, self.model, output, self.num_augmentations, device)
            score = f.JSD(aug_out, SPACING32).cpu()
            assert score.shape[0] == img.shape[0], "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack([torch.arange(offset, offset + img.shape[0]), score]).data
            q.extend(idx_entropy)

        return q.data


class BALD(QueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader)
        assert "trainer" in kwargs
        self.model = kwargs["trainer"].model
        self.dropout_round = int(kwargs.get("round", 10))
        assert hasattr(self.model, "dropout_switch")

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        self.model.dropout_switch(True)

        device = next(iter(self.model.parameters())).device
        q = LimitSortedList(limit=query_num, descending=True)
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            out = torch.empty(self.dropout_round, img.shape[0], 2, img.shape[-2], img.shape[-1], device=device)
            for round_ in range(self.dropout_round):
                img = img.to(device)
                output = self.model(img).softmax(dim=1)
                out[round_] = output
            score = f.JSD(out, SPACING32).cpu()
            assert score.shape[0] == img.shape[0], "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack([torch.arange(offset, offset + img.shape[0]), score]).data
            q.extend(idx_entropy)

        return q.data


class LossPredictionQuery(SimpleQueryStrategy):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=True, **kwargs)

    def compute_score(self, model_output):
        self.trainer.loss_predition_module.eval()
        _, features = model_output
        pred_loss = self.trainer.loss_predition_module(features)
        return pred_loss


class CoresetQuery(QueryStrategy):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model

        pool_size = int(kwargs.get("pool_size", 16))
        self.pool = nn.MaxPool2d(pool_size)
        self.pool.eval()

    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            _, features = self.model(img)
            embedding = self.pool(features[-1]).view((img.shape[0], -1))
            embedding_list.append(embedding)
        return torch.concat(embedding_list, dim=0)

    def select_dataset_idx(self, query_num):
        self.model.eval()
        embedding_unlabeled = self.embedding(self.unlabeled_dataloader)
        embedding_labeled = self.embedding(self.labeled_dataloader)
        return self.furthest_first(unlabeled_set=embedding_unlabeled.cpu().numpy(),
                                   labeled_set=embedding_labeled.cpu().numpy(),
                                   budget=query_num)

    def furthest_first(self, unlabeled_set, labeled_set, budget):
        """
        Selects points with maximum distance

        Parameters
        ----------
        unlabeled_set: numpy array
            Embeddings of unlabeled set
        labeled_set: numpy array
            Embeddings of labeled set
        budget: int
            Number of points to return
        Returns
        ----------
        idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """
        m = np.shape(unlabeled_set)[0]
        if np.shape(labeled_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(unlabeled_set, labeled_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(budget):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(
                unlabeled_set, unlabeled_set[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs
