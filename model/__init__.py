from .Unet import UNet,UNetWithDropout,UNetWithFeature
import random
import numpy as np
import torch.nn as nn
def get_rampup_ratio(i, start, end, mode = "linear"):
    """
    Obtain the rampup ratio.
    :param i: (int) The current iteration.
    :param start: (int) The start iteration.
    :param end: (int) The end itertation.
    :param mode: (str) Valid values are {`linear`, `sigmoid`, `cosine`}.
    """
    i = np.clip(i, start, end)
    if(mode == "linear"):
        rampup = (i - start) / (end - start)
    elif(mode == "sigmoid"):
        phase  = 1.0 - (i - start) / (end - start)
        rampup = float(np.exp(-5.0 * phase * phase))
    elif(mode == "cosine"):
        phase  = 1.0 - (i - start) / (end - start)
        rampup = float(.5 * (np.cos(np.pi * phase) + 1))
    else:
        raise ValueError("Undefined rampup mode {0:}".format(mode))
    return rampup

def initialize_weights(param, p):

    class_name = param.__class__.__name__
    if class_name.startswith('Conv2d') and random.random() <= p:
        # Initialization according to original Unet paper
        # See https://arxiv.org/pdf/1505.04597.pdf
        _, in_maps, k, _ = param.weight.shape
        n = k * k * in_maps
        std = np.sqrt(2 / n)
        nn.init.normal_(param.weight.data, mean=0.0, std=std)


def build_model(model: str):
    model = model.lower()
    if model == "unet":
        return UNet(1, 2, 16)
    elif model == "unet32":
        return UNet(1, 2, 32)
    elif model == "unet48":
        return UNet(1, 2, 48)
    elif model == "unet64":
        return UNet(1, 2, 48)
    else:
        raise NotImplementedError
