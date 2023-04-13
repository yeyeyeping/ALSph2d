import torch
import numpy as np


@torch.jit.script
def margin_confidence(model_output: torch.Tensor) -> torch.Tensor:
    return torch.abs(model_output[:, 0] - model_output[:, 1]).mean(dim=(-1, -2))


@torch.jit.script
def least_confidence(model_output: torch.Tensor) -> torch.Tensor:
    output_max = torch.max(model_output, dim=1)[0]
    return output_max.mean(dim=(-2, -1))


@torch.jit.script
def max_entropy(model_output: torch.Tensor, spacing32: float) -> torch.Tensor:
    return torch.mean(-model_output * torch.log(model_output + spacing32), dim=(1, 2, 3))


@torch.jit.script
def JSD(data: torch.Tensor, spacing32: float) -> torch.Tensor:
    # data:round x batch x class x height x weight
    mean = data.mean(0)
    # mean entropy per pixel
    mean_entropy = -torch.mean(mean * torch.log(mean + spacing32), dim=[-3, -2, -1])
    sample_entropy = -torch.mean(torch.mean(data * torch.log(data + spacing32), dim=[-3, -2, -1]), dim=0)
    return mean_entropy - sample_entropy
