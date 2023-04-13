import numpy as np
import torch
from torch.utils.data import DataLoader
import util.jitfunc as f

from util.taalhelper import augments_forward

SPACING32: float = np.spacing(1, dtype=np.float32)

class LimitSortedList(object):

    def __init__(self, limit, descending=False) -> None:
        self.descending = descending
        self.limit = limit
        self._data = []

    def reset(self):
        self._data.clear()

    @property
    def data(self):
        return list(map(lambda x: int(x[0]), self._data))

    def extend(self, idx):
        assert isinstance(idx, (torch.Tensor, np.ndarray, list, tuple))
        idx = list(idx)
        self._data.extend(idx)
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


class RandomQuery(QueryStrategy):
    def sample(self, query_num):
        np.random.shuffle(self.unlabeled_dataloader.sampler.indices)
        self.labeled_dataloader.sampler.indices.extend(self.unlabeled_dataloader.sampler.indices[:query_num])
        del self.unlabeled_dataloader.sampler.indices[:query_num]


class SimpleQueryStrategy(QueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader)
        assert "model" in kwargs
        assert "descending" in kwargs
        self.model = kwargs["model"]
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
            output = self.model(img).softmax(dim=1)
            score = self.compute_score(output).cpu()
            assert score.shape[0] == img.shape[0], "shape dismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entopy = torch.column_stack([torch.arange(offset, offset + img.shape[0]), score]).data
            q.extend(idx_entopy)
        return q.data


class MaxEntropy(SimpleQueryStrategy):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=True, **kwargs)

    def compute_score(self, model_output):
        return f.max_entropy(model_output, SPACING32)


class MarginConfidence(SimpleQueryStrategy):

    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        return f.margin_confidence(model_output)


class LeastConfidence(SimpleQueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        output_max = torch.max(model_output, dim=1)[0]
        return output_max.mean(dim=(-2, -1))


class TAAL(QueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader)
        assert "model" in kwargs
        self.model = kwargs["model"]
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
            assert score.shape[0] == img.shape[0], "shape dismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entopy = torch.column_stack([torch.arange(offset, offset + img.shape[0]), score]).data
            q.extend(idx_entopy)

        return q.data


class BALD(QueryStrategy):
    def __init__(self, unlabeled_dataloader: DataLoader, labeled_dataloader: DataLoader, **kwargs) -> None:
        super().__init__(unlabeled_dataloader, labeled_dataloader)
        assert "model" in kwargs
        self.model = kwargs["model"]
        self.dropout_round = int(kwargs.get("round", 10))
        assert hasattr(self.model, "dropout_switch")

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        self.model.dropout_switch(True)

        device = next(iter(self.model.parameters())).device
        q = LimitSortedList(limit=query_num, descending=False)
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            out = torch.empty(self.dropout_round, img.shape[0], 2, img.shape[-2], img.shape[-1], device=device)
            for round in range(self.dropout_round):
                img = img.to(device)
                output = self.model(img).softmax(dim=1)
                out[round] = output
            score = f.mutual_information(out, SPACING32).cpu()
            assert score.shape[0] == img.shape[0], "shape dismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entopy = torch.column_stack([torch.arange(offset, offset + img.shape[0]), score]).data
            q.extend(idx_entopy)

        return q.data
