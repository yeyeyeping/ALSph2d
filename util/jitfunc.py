import torch


@torch.jit.script
def margin_confidence(model_output: torch.Tensor, weight=torch.Tensor([1])) -> torch.Tensor:
    if weight.device != model_output.device:
        weight = weight.to(model_output.device)
    weight_socre = torch.abs(model_output[:, 0] - model_output[:, 1]) * weight
    return weight_socre.mean(dim=(-1, -2))


@torch.jit.script
def least_confidence(model_output: torch.Tensor, weight=torch.Tensor([1])) -> torch.Tensor:
    if weight.device != model_output.device:
        weight = weight.to(model_output.device)
    output_max = torch.max(model_output, dim=1)[0] * weight
    return output_max.mean(dim=(-2, -1))


@torch.jit.script
def max_entropy(model_output: torch.Tensor, weight=torch.Tensor([1])) -> torch.Tensor:
    if weight.device != model_output.device:
        weight = weight.to(model_output.device)
    weight_score = -model_output * torch.log(model_output + 1e-7)
    return torch.mean(weight_score.mean(1) * weight, dim=(-2, -1))


def hisgram_entropy(model_output: torch.Tensor, weight=torch.Tensor([1])):
    if weight.device != model_output.device:
        weight = weight.to(model_output.device)
    score = []
    for output in weight:
        frequency, _ = torch.histogram(output, bins=10)
        probs = frequency / frequency.sum()
        entropy = torch.nansum(-probs * torch.log(probs + 1e-7))
        score.append(entropy)
    return torch.tensor(score)


# @torch.jit.script
def JSD(data: torch.Tensor) -> torch.Tensor:
    # data:round x batch x class x height x width
    mean = data.mean(0)
    # mean entropy per pixel
    mean_entropy = -torch.mean(mean * torch.log(mean + 1e-7), dim=[-3, -2, -1])
    sample_entropy = -torch.mean(torch.mean(data * torch.log(data + 1e-7), dim=[-3, -2, -1]), dim=0)
    return mean_entropy - sample_entropy


if __name__ == '__main__':
    import numpy as np

    a, b = torch.randn(size=(16, 2, 16, 16)), torch.randn(size=(16, 16, 16),device="cuda")
    print(hisgram_entropy(a.softmax(1), b))
