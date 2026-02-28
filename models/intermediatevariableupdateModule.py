import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFM(nn.Module):  # Parallel cross-fusion module
    def __init__(self):
        super(PFM, self).__init__()

        self.convf = nn.Conv2d(8, 32, kernel_size=3, padding=1)  # Map f to 32 channels
        self.convh = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Map h to 32 channels
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.convf_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.convh_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.gate = nn.LSTM(64, 32, batch_first=True)  # Select complementary features

    def forward(self, h, f):
        # print("f shape", f.shape)   # [1,8,400,400]
        f = F.relu(self.convf(f))  # Map f to 32 channels
        # print(f.shape)
        h = F.relu(self.convh(h))  # Map h to 32 channels
        f_s = self.sigmoid(f)  # Element-wise addition with Sigmoid activation
        h_s = self.sigmoid(h)
        h_m = h * f_s
        f_m = f * h_s
        f_p = f_m + f
        h_p = h_m + h
        f = self.convf_1(f_p)
        h = self.convh_1(h_p)
        f = self.sigmoid(f)
        h = self.tanh(h)
        # f, _ = self.gate(f.unsqueeze(0))  # Use gating mechanism to select complementary features
        final = f * h
        return final  # 32 channels


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        self.inter_channels = in_channels // 2
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1)

        self.conv_out = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)
        theta = theta.view(theta.size(0), self.inter_channels, -1)
        phi = phi.view(phi.size(0), self.inter_channels, -1).permute(0, 2, 1)
        g = g.view(g.size(0), self.inter_channels, -1).permute(0, 2, 1)

        f = torch.matmul(theta, phi)
        f_div_c = self.softmax(f)

        f_div_c = f_div_c.permute(0, 2, 1)
        g = g.permute(0, 2, 1)
        # print("f_div_c shape:", f_div_c.shape)
        # print("g shape:", g.shape)
        y = torch.matmul(f_div_c, g)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(y.size(0), self.inter_channels, *x.size()[2:])
        y = self.conv_out(y)
        output = x + y
        return output


class MultiHeadSpatialAttention(nn.Module):
    """Multi-Head Self-Attention for 2D feature maps.
    
    Replaces LSTM with parallel attention mechanism for better efficiency
    and spatial structure preservation.
    """
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadSpatialAttention, self).__init__()
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.inner_dim = self.head_dim * num_heads
        
        # Q, K, V projections using 1x1 convolutions
        self.to_qkv = nn.Conv2d(in_channels, self.inner_dim * 3, kernel_size=1, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Conv2d(self.inner_dim, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Generate Q, K, V
        qkv = self.to_qkv(x)  # (b, inner_dim*3, h, w)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, b, heads, h*w, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, heads, h*w, h*w)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (b, heads, h*w, head_dim)
        
        # Reshape back to spatial
        out = out.permute(0, 1, 3, 2)  # (b, heads, head_dim, h*w)
        out = out.reshape(b, self.inner_dim, h, w)
        
        # Output projection with residual connection
        return x + self.to_out(out)


class EGIM(nn.Module):
    def __init__(self):
        super(EGIM, self).__init__()

        # self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.baseBlock = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.pfm = PFM()
        self.nonlocalblock = NonLocalBlock(32)

    def forward(self, h, f):
        pfm = self.pfm(h, f)
        bb_1 = self.baseBlock(pfm)
        output_up = self.nonlocalblock(bb_1)
        # -----------LSTM-------------------------------------------------------------------------
        # Get dimensions of input LSTM image
        # batch_size, channel, height, width = bb_1.shape
        # # print(bb_1.shape)
        # target_size = bb_1.shape[2:]
        # # Set feature dimension (sequence length for LSTM initialization)
        # input_size = channel * width
        # # Initialize parameter list, second param is hidden_size, third is num LSTM layers
        # lstm = nn.LSTM(input_size, 256, 2)
        # # Image height or width as time step dimension
        # seq_dim = height
        # # Permute dimensions
        # bb_1_permuted = bb_1.permute(0, 2, 1, 3)
        # bb_1_reshaped = bb_1_permuted.reshape(batch_size * seq_dim, channel, width)
        # # Reshape image to LSTM input shape
        # seq_length = batch_size * seq_dim
        # input_size = channel * width
        # lstm_input = bb_1_reshaped.reshape(seq_length, batch_size, input_size)
        # lstm = lstm.to(lstm_input.device)  # Move all params to same device as input
        # output, _ = lstm(lstm_input)
        # # print(output.shape)   # [width,1,256]
        # # print("test_only", test_only)
        # if not self.training:
        #     maxpool = nn.MaxPool1d(kernel_size=2)
        #     output = maxpool(output)
        #     output_width, output_batch, output_len = output.shape
        #     # output_batch = output.shape[1]
        #     # output_len = output.shape[2]
        #     sum = output_len * output_batch * output_width
        #     output_shape = sum // channel
        #     output_shape = math.sqrt(output_shape)
        #     output_shape = int(output_shape)
        # else:
        #     # print("Executed else branch")
        #     output_shape = 20
        # output_reshaped = output.view(batch_size, channel, output_shape, output_shape)
        # # print(output_reshaped.shape)
        # output_up = F.interpolate(output_reshaped, size=target_size, mode='bilinear',
        #                           align_corners=False)
        # -------------------------------------------------------------------------------------
        bb_2 = self.baseBlock(output_up)
        bb_3 = self.conv2(bb_2)
        final = bb_3 + h

        return final
