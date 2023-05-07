from typing import NewType

import scipy.ndimage as ndimage
import numpy as np


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


def build_strategy(strategy: str):
    import util.query_strategy as qs
    from util.trainer import BaseTrainer, TTATrainer, BALDTrainer, LearningLossTrainer, CoresetTrainer, \
        ConstrativeTrainer, DEALTrainer
    from util.query_strategy import TAAL, BALD, LossPredictionQuery, CoresetQuery, ConstrativeQuery, DEALQuery

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
    elif strategy == ConstrativeQuery:
        trainer = ConstrativeTrainer
    elif strategy == DEALQuery:
        trainer = DEALTrainer
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


if __name__ == '__main__':
    print(build_strategy("MaxEntropy"))
