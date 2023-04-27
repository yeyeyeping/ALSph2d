# unet decoder feature
# torch.Size([16, 384, 52, 52])
# torch.Size([16, 192, 104, 104])
# torch.Size([16, 96, 208, 208])
# torch.Size([16, 48, 416, 416])
import torch
import torch.nn as nn
from torch.nn import functional as F


class LossPredModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.GAP0 = nn.AvgPool2d(26)
        self.GAP1 = nn.AvgPool2d(26)
        self.GAP2 = nn.AvgPool2d(26)
        self.GAP3 = nn.AvgPool2d(32)

        self.FC0 = nn.Linear(384 * 4, 128)
        self.FC1 = nn.Linear(192 * 16, 256)
        self.FC2 = nn.Linear(96 * 64, 512)
        self.FC3 = nn.Linear(48 * 169, 1024)

        self.linear = nn.Linear(128 + 256 + 512 + 1024, 128)
        self.linearf = nn.Linear(128, 1)

    def forward(self, features):
        out1 = self.GAP0(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC0(out1))

        out2 = self.GAP1(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC1(out2))

        out3 = self.GAP2(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC2(out3))

        out4 = self.GAP3(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC3(out4))

        out = self.linearf(F.relu(self.linear(torch.cat((out1, out2, out3, out4), 1))))
        return out


# LossPredictionLoss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    if len(input) % 2 == 0:
        input = input[:-1]
        target = target[:-1]
    # assert len(input) % 2 == 0, f'the batch size {len(input)}is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], batch size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already haved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


if __name__ == '__main__':
    feature = []
    feature.append(torch.randn([16, 384, 52, 52], device="cuda"))
    feature.append(torch.randn([16, 192, 104, 104], device="cuda"))
    feature.append(torch.randn([16, 96, 208, 208], device="cuda"))
    feature.append(torch.randn([16, 48, 416, 416], device="cuda"))
    model = LossPredModule().cuda()
    print(model(feature))
