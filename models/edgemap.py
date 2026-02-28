import torch
import torch.nn.functional as F
from torch import nn


class EdgeMap(nn.Module):
    def __init__(self, num_channels=3):
        super(EdgeMap, self).__init__()

        # Define 8 gradient convolution kernels
        # Horizontal Sobel operator kernel
        # unsqueeze expands dimension at specified position, used twice to convert 2D (H,W) to 4D (B,C,H,W)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # repeat duplicates data along specified dimension for per-channel convolution
        self.sobel_x = self.sobel_x.repeat(1, num_channels, 1, 1)
        self.sobel_x = self.sobel_x.cuda()

        # Vertical Sobel operator kernel
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = self.sobel_y.repeat(1, num_channels, 1, 1)
        self.sobel_y = self.sobel_y.cuda()

        # Horizontal Prewitt operator kernel
        self.prewitt_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        self.prewitt_x = self.prewitt_x.repeat(1, num_channels, 1, 1)

        self.prewitt_x = self.prewitt_x.cuda()

        # Vertical Prewitt operator kernel
        self.prewitt_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        self.prewitt_y = self.prewitt_y.repeat(1, num_channels, 1, 1)
        self.prewitt_y = self.prewitt_y.cuda()

        # Sobel main diagonal edge kernel
        self.sobel_d1 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        self.sobel_d1 = self.sobel_d1.repeat(1, num_channels, 1, 1)
        self.sobel_d1 = self.sobel_d1.cuda()

        # Sobel anti-diagonal edge kernel
        self.sobel_d2 = torch.tensor([[-2, -1, 0], [1, 0, -1], [0, 1, 2]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        self.sobel_d2 = self.sobel_d2.repeat(1, num_channels, 1, 1)
        self.sobel_d2 = self.sobel_d2.cuda()

        # Prewitt main diagonal edge kernel
        self.prewitt_d1 = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        self.prewitt_d1 = self.prewitt_d1.repeat(1, num_channels, 1, 1)
        self.prewitt_d1 = self.prewitt_d1.cuda()

        # Prewitt anti-diagonal edge kernel
        self.prewitt_d2 = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        self.prewitt_d2 = self.prewitt_d2.repeat(1, num_channels, 1, 1)
        self.prewitt_d2 = self.prewitt_d2.cuda()

    def forward(self, x):
        # Compute 8 convolution results

        sobel_x = F.conv2d(x, self.sobel_x, padding=1)
        sobel_y = F.conv2d(x, self.sobel_y, padding=1)
        prewitt_x = F.conv2d(x, self.prewitt_x, padding=1)
        prewitt_y = F.conv2d(x, self.prewitt_y, padding=1)
        sobel_d1 = F.conv2d(x, self.sobel_d1, padding=1)
        sobel_d2 = F.conv2d(x, self.sobel_d2, padding=1)
        prewitt_d1 = F.conv2d(x, self.prewitt_d1, padding=1)
        prewitt_d2 = F.conv2d(x, self.prewitt_d2, padding=1)

        # Concatenate 8 results into an 8-channel tensor
        edge = torch.cat((sobel_x, sobel_y, prewitt_x, prewitt_y, sobel_d1, sobel_d2, prewitt_d1, prewitt_d2), dim=1)

        return edge
