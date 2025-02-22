import torch
import torch.nn as nn


class GSA(nn.Module):
    def __init__(self, c1, c2, g=2):

        super(GSA, self).__init__()
        self.g = g
        self.pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, None, None)))
        self.cv = nn.Conv2d(c1, 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        short = x
        x = x.unsqueeze(1)
        y = list(x.chunk(self.g, 2))
        y1 = [self.pool(y) for y in y]
        y2 = [torch.randn(1, 1, 1, 1, 1)] * self.g
        for i in range(0, self.g):
            # y[i] = pool(y[i])
            y2[i] = y[i] * torch.sigmoid(y1[i])
        y_cat = torch.cat(y2, 2)
        y_cat = y_cat.squeeze(1)
        y3 = self.bn(self.cv(y_cat))

        return torch.sigmoid(y3) * short


if __name__ == '__main__':

    ip = torch.randn(1, 64, 8, 9)
    y0 = ip.unsqueeze(1)
    y = list(y0.chunk(4, 2))
    pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, None, None)))
    y1 = [pool(y) for y in y]
    y2 = [torch.randn(1, 1, 1, 1, 1)] * 4
    for i in range(0, 4):
        # y[i] = pool(y[i])
        y2[i] = y[i] * y1[i]
    y_cat = torch.cat(y2, 2)
    # y_cat = y_cat.squeeze(1)

    print(y_cat[-1].size())
