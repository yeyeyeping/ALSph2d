from typing import NewType
from dataset.SphDataset import SubsetSampler
import torch
import scipy.ndimage as ndimage
import numpy as np
import albumentations as A
from dataset.SphDataset import Dataset2d, Dataset3d
from os.path import join


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    def reset(self):
        self.initialized = False

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 3)


def save_query_plot(folder, labeled_percent, dice_list):
    import matplotlib.pyplot as plt
    with open(f"{folder}/result.txt", "w") as fp:
        fp.write("x:")
        fp.write(str(labeled_percent))
        fp.write("\ny:")
        fp.write(str(dice_list))
    plt.plot(labeled_percent, dice_list)
    plt.savefig(f"{folder}/result.jpg")


def get_samplers(data_num, initial_labeled, with_pseudo=False):
    data_indice = list(range(data_num))
    np.random.shuffle(data_indice)
    retval = (SubsetSampler(data_indice[:initial_labeled]), SubsetSampler(data_indice[initial_labeled:]))
    if with_pseudo:
        retval = (*retval, SubsetSampler([]))
    return retval


def build_strategy(strategy: str):
    import util.query_strategy as qs
    from util.trainer import BaseTrainer, TTATrainer, BALDTrainer, LearningLossTrainer, CoresetTrainer, \
        ContrastiveTrainer, DEALTrainer, URPCTrainer, OnlineMGTrainer
    from util.query_strategy import TAAL, BALD, LossPredictionQuery, CoresetQuery, ContrastiveQuery, DEALQuery, \
        OnlineMGQuery

    if strategy in qs.__dict__:
        strategy = qs.__dict__[strategy]
    else:
        raise NotImplementedError
    trainer = BaseTrainer

    if strategy == TAAL:
        trainer = TTATrainer
    elif strategy == BALD:
        trainer = BALDTrainer
    elif strategy == LossPredictionQuery:
        trainer = LearningLossTrainer
    elif strategy == CoresetQuery:
        trainer = CoresetTrainer
    elif strategy == ContrastiveQuery:
        trainer = ContrastiveTrainer
    elif strategy == DEALQuery:
        trainer = DEALTrainer
    elif strategy == OnlineMGQuery:
        trainer = OnlineMGTrainer

    return strategy, trainer


def get_largest_k_components(image, k=1):
    """
    Get the largest K components from 2D or 3D binary image.

    :param image: The input ND array for binary segmentation.
    :param k: (int) The value of k.

    :return: An output array with only the largest K components of the input.
    """
    dim = len(image.shape)
    if (image.sum() == 0):
        print('the largest component is null')
        return image
    if (dim < 2 or dim > 3):
        raise ValueError("the dimension number should be 2 or 3")
    s = ndimage.generate_binary_structure(dim, 1)
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    sizes_sort = sorted(sizes, reverse=True)
    kmin = min(k, numpatches)
    output = np.zeros_like(image)
    for i in range(kmin):
        labeli = np.where(sizes == sizes_sort[i])[0] + 1
        output = output + np.asarray(labeled_array == labeli, np.uint8)
    return output


def label_smooth(volume):
    [D, H, W] = volume.shape
    s = ndimage.generate_binary_structure(2, 1)
    for d in range(D):
        if (volume[d].sum() > 0):
            volume_d = get_largest_k_components(volume[d], k=5)
            if (volume_d.sum() < 10):
                volume[d] = np.zeros_like(volume[d])
                continue
            volume_d = ndimage.morphology.binary_closing(volume_d, s)
            volume_d = ndimage.morphology.binary_opening(volume_d, s)
            volume[d] = volume_d
    return volume


def get_dataloader(args, with_pseudo=False):
    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.2),
    ])
    dataset_train, dataset_val = Dataset2d(datafolder=join(args.data_dir, "train"),
                                           transform=train_transform), \
        Dataset3d(folder=join(args.data_dir, "test"))
    labeled_sampler, *unlabeled_sampler = get_samplers(len(dataset_train), args.initial_labeled,
                                                       with_pseudo=with_pseudo)
    retval = {}
    if with_pseudo:
        from dataset.SphDataset import PseudoDataset2d
        unlabeled_sampler, pseudo_sampler = unlabeled_sampler
        # dpseudo = torch.utils.data.DataLoader(PseudoDataset2d(datafolder=join(args.data_dir, "train"),
        #                                                       transform=train_transform),
        #                                       batch_size=args.batch_size,
        #                                       sampler=pseudo_sampler)
        dpseudo = torch.utils.data.DataLoader(PseudoDataset2d(datafolder=join(args.data_dir, "train"),
                                                              transform=train_transform),
                                              batch_size=args.batch_size,
                                              sampler=pseudo_sampler,
                                              persistent_workers=True,
                                              pin_memory=True,
                                              prefetch_factor=4,
                                              num_workers=args.num_workers)
        retval["pseudo"] = dpseudo
    else:
        unlabeled_sampler = unlabeled_sampler[0]

    dulabeled = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=args.batch_size,
                                            sampler=unlabeled_sampler,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            prefetch_factor=4,
                                            num_workers=args.num_workers)

    dlabeled = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=args.batch_size,
                                           sampler=labeled_sampler,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           prefetch_factor=4,
                                           num_workers=args.num_workers)

    dval = torch.utils.data.DataLoader(dataset_val,
                                       batch_size=1,
                                       persistent_workers=True,
                                       pin_memory=True,
                                       prefetch_factor=4,
                                       num_workers=args.num_workers)

    return {
        **retval,
        "labeled": dlabeled,
        "unlabeled": dulabeled,
        "test": dval
    }



def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 80)

if __name__ == '__main__':
    print(build_strategy("MaxEntropy"))
