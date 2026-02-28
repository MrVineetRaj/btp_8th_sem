import torch
import torch.nn as nn


class EAFM(nn.Module):
    def __init__(self):
        super(EAFM, self).__init__()
        # First Conv layer
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        # First BaseBlock
        self.baseblock = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        # Second BaseBlock
        # self.baseblock2 = nn.Sequential(
        #     nn.Conv2d(8, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        # )

    def forward(self, z):
        # First Conv layer
        x = self.conv1(z)
        # First BaseBlock
        out1 = x + self.baseblock(x)
        # Second BaseBlock
        out2 = out1 + self.baseblock(out1)
        # Second Conv layer
        out3 = self.conv2(out2)
        # Residual connection 2
        out3 += z
        # Return updated feature map
        return out3
