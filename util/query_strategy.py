import random
from pymic.util.evaluation_seg import binary_dice
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.jitfunc as f
from util.taalhelper import augments_forward
from torch import nn
from sklearn.metrics import pairwise_distances
from torch.nn import functional as F


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

    def extend(self, idx_score):
        assert isinstance(idx_score, (torch.Tensor, np.ndarray, list, tuple))
        idx_score = list(idx_score)
        self._data.extend(idx_score)
        if len(self._data) > self.limit:
            self._data = sorted(self._data, key=lambda x: x[1], reverse=self.descending)[:self.limit]


class QueryStrategy(object):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
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
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack([torch.arange(offset, offset + len(img)), score])
            q.extend(idx_entropy)
        return q.data


class RandomQuery(QueryStrategy):
    def sample(self, query_num):
        np.random.shuffle(self.unlabeled_dataloader.sampler.indices)
        self.labeled_dataloader.sampler.indices.extend(self.unlabeled_dataloader.sampler.indices[:query_num])
        del self.unlabeled_dataloader.sampler.indices[:query_num]


class MaxEntropy(SimpleQueryStrategy):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=True, **kwargs)

    def compute_score(self, model_output):
        model_output, _ = model_output
        return f.max_entropy(model_output.softmax(dim=1))


class MarginConfidence(SimpleQueryStrategy):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        model_output, _ = model_output
        return f.margin_confidence(model_output.softmax(dim=1))


class LeastConfidence(SimpleQueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        model_output, _ = model_output
        return f.least_confidence(model_output.softmax(dim=1))


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
            output, _ = self.model(img)
            aug_out = augments_forward(img, self.model, output.softmax(dim=1), self.num_augmentations, device)
            score = f.JSD(aug_out).cpu()
            assert score.shape[0] == img.shape[0], "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack([torch.arange(offset, offset + img.shape[0]), score])
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
                output, _ = self.model(img)
                out[round_] = output.softmax(dim=1)
            score = f.JSD(out).cpu()
            assert score.shape[0] == img.shape[0], "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack([torch.arange(offset, offset + img.shape[0]), score])
            q.extend(idx_entropy)

        return q.data


class LossPredictionQuery(SimpleQueryStrategy):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=True, **kwargs)

    @torch.no_grad()
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

        pool_size = int(kwargs.get("pool_size", 12))
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.pool.eval()

    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            _, features = self.model(img)
            embedding = self.pool(features[0]).view((img.shape[0], -1))
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

        for _ in range(budget):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(
                unlabeled_set, unlabeled_set[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs


class UncertaintyBatchQuery(QueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader)
        assert "trainer" in kwargs
        self.model = kwargs["trainer"].model

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        uncertainty_score = []
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(device)
            output, _ = self.model(img)
            score = f.max_entropy(output.softmax(dim=1)).cpu()
            assert score.shape[0] == img.shape[0], "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack([torch.arange(offset, offset + img.shape[0]), score])
            uncertainty_score.extend(idx_entropy.tolist())
        random.shuffle(uncertainty_score)
        # todo:Is that  necessary to drop last few samplers?
        splits = []
        for s in range(0, len(uncertainty_score), query_num):
            end = s + query_num
            if end >= len(uncertainty_score):
                end = None
            splits.append(uncertainty_score[s: end])
        batch_uncertainty = list(map(lambda x: np.sum(x, axis=0)[1], splits))
        max_idx = np.argmax(batch_uncertainty)
        selected_batch = np.asarray(splits[max_idx])
        return selected_batch[:, 0].astype(np.uint64)


class ContrastiveQuery(QueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader)
        assert "trainer" in kwargs
        self.model = kwargs["trainer"].model
        self.k = int(kwargs.get("constrative_sampler_size", 20))
        self.distance_measure = kwargs.get("distance_measure", "cosine")
        assert self.k < len(
            unlabeled_dataloader.sampler.indices), f"{self.k} > {len(labeled_dataloader.sampler.indices)}"
        pool_size = int(kwargs.get("pool_size", 12))
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.pool.eval()

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        labeled_feature_list, unlabeled_feature_list = [], []
        for _, (img, _) in enumerate(self.labeled_dataloader):
            img = img.to(device)
            _, features = self.model(img)
            embedding = self.pool(features[0]).view((img.shape[0], -1))
            normal_feature = F.normalize(embedding, dim=1)
            labeled_feature_list.append(normal_feature.cpu().numpy())
        labeled_feature = np.concatenate(labeled_feature_list)

        for _, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(device)
            _, features = self.model(img)
            embedding = self.pool(features[0]).view((img.shape[0], -1))
            normal_feature = F.normalize(embedding, dim=1)
            unlabeled_feature_list.append(normal_feature.cpu().numpy())
        unlabeled_feature = np.concatenate(unlabeled_feature_list)

        distances = pairwise_distances(unlabeled_feature, labeled_feature, metric=self.distance_measure)

        del unlabeled_feature, labeled_feature

        argidx = np.argsort(distances, axis=1)
        kidx = argidx[:, :self.k]
        shape = kidx.shape
        dataset_idx = np.asarray(self.convert2img_idx(kidx.flatten(), self.labeled_dataloader), dtype=np.int32).reshape(
            shape)
        labeled_dataset, unlabeled_dataset = self.labeled_dataloader.dataset, self.unlabeled_dataloader.dataset
        q = []

        for unlabeled_idx, labeled_idxs in enumerate(dataset_idx):
            unlabeled_img, _ = unlabeled_dataset[self.unlabeled_dataloader.sampler.indices[unlabeled_idx]]
            labeled_img, _ = zip(*[labeled_dataset[img_idx] for img_idx in labeled_idxs])
            labeled_img = torch.stack(labeled_img)
            unlabeled_img, labeled_img = unlabeled_img.to(device), labeled_img.to(device)
            unlab_pred, _ = self.model(unlabeled_img.unsqueeze(0))
            unlab_pred = unlab_pred.softmax(1).repeat(labeled_img.shape[0], 1, 1, 1)[:, 1]
            lab_pred, _ = self.model(labeled_img)
            lab_pred = lab_pred.softmax(1)[:, 1]
            q.append((unlabeled_idx, binary_dice(lab_pred.cpu().numpy(), unlab_pred.cpu().numpy())))

        torch.cuda.empty_cache()

        return map(lambda x: x[0], sorted(q, key=lambda x: x[1])[:query_num])


class DEALQuery(SimpleQueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        funcstr = kwargs.get("difficulty_strategy", "max_entropy")
        assert funcstr in f.__dict__, f"{funcstr} not implement"
        if funcstr in ("max_entropy", "hisgram_entropy"):
            super().__init__(unlabeled_dataloader, labeled_dataloader, descending=True, **kwargs)
        else:
            super().__init__(unlabeled_dataloader, labeled_dataloader, descending=False, **kwargs)
        self.score_func = f.__dict__[funcstr]

    @torch.no_grad()
    def compute_score(self, model_output):
        model_output, _ = model_output
        self.trainer.pam.eval()
        difficulty_map, _ = self.trainer.pam(model_output)
        return self.score_func(model_output.softmax(1), difficulty_map)
