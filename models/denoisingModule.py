import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            # Conv2d input/output sizes are first two parameters
            m.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):  # x(n_feat) -> res(n_feat)
        res = self.body(x).mul(self.res_scale)  # Scale all elements by residual scale
        res += x
        return res


class EncodingBlock(nn.Module):
    def __init__(self, ch_in):
        super(EncodingBlock, self).__init__()

        body = [
            nn.Conv2d(ch_in, 64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            nn.Conv2d(64, 128, kernel_size=3, padding=3 // 2)
        ]
        self.body = nn.Sequential(*body)
        self.down = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=3 // 2)
        self.act = nn.ReLU()

    def forward(self, input):  # input -> f_e(128), down(64)
        f_e = self.body(input)
        down = self.act(self.down(f_e))
        return f_e, down  # f_e contains richer high-level features, down is coarse downsampled features


class EncodingBlockEnd(nn.Module):
    def __init__(self, ch_in):
        super(EncodingBlockEnd, self).__init__()

        head = [
            nn.Conv2d(in_channels=ch_in, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU()
        ]
        body = [
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),

            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
        ]
        tail = [
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)
        ]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, input):  # input -> f_e(128)
        out = self.head(input)
        f_e = self.body(out) + out
        f_e = self.tail(f_e)
        return f_e


class DecodingBlock(nn.Module):
    def __init__(self, ch_in):
        super(DecodingBlock, self).__init__()

        body = [
            nn.Conv2d(in_channels=ch_in, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=1 // 2)
        ]

        self.up = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()
        self.body = nn.Sequential(*body)

    def forward(self, input, map):  # input(128), map(128) -> out(256)
        # Ensure transposed conv output shape matches map shape
        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        up = self.act(up)
        out = torch.cat((up, map), 1)  # Concatenate along channel dimension
        out = self.body(out)
        return out


# The only difference from previous decoding is an extra conv layer at the end to increase channels
class DecodingBlockEnd(nn.Module):
    def __init__(self, ch_in):
        super(DecodingBlockEnd, self).__init__()

        body = [
            nn.Conv2d(ch_in, 64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            ResBlock(n_feat=64, kernel_size=3),
            ResBlock(n_feat=64, kernel_size=3),
        ]

        self.up = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()
        self.body = nn.Sequential(*body)

    def forward(self, input, map):  # input(128), map(128) -> out(64)
        # Ensure transposed conv output shape matches map shape
        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        out = self.act(up)
        out = torch.cat((out, map), 1)  # Concatenate along channel dimension
        out = self.body(out)
        return out
