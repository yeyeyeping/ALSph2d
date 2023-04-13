from .Unet import UNet,UNetWithDropout
import random
import numpy as np
import torch.nn as nn

def initialize_weights(param, p):

    class_name = param.__class__.__name__
    if class_name.startswith('Conv') and random.random() <= p:
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
    elif model == "unetwithdropout":
        return UNetWithDropout(1, 2, 16)
    else:
        raise NotImplementedError
