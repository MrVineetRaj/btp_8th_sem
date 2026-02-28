import torch
import torch.nn as nn
import torch.nn.functional as F

from models.intermediatevariableupdateModule import NonLocalBlock, MultiHeadSpatialAttention

class PFM_1(nn.Module):  # 并行交叉融合模块
    def __init__(self):
        super(PFM_1, self).__init__()

        self.convh = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 将h映射为32通道
        self.convr = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 将r映射为32通道
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.convh_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.convr_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.gate = nn.LSTM(64, 32, batch_first=True)  # 选取互补特征

    def forward(self, r, h):
        # print("h的shape", h.shape)
        h = F.relu(self.convh(h))  # 将h映射为32通道
        # print(h.shape)
        r = F.relu(self.convr(r))  # 将h映射为32通道
        h_s = self.sigmoid(h)  # 以元素方式添加并Sigmoid激活
        r_s = self.sigmoid(r)
        r_m = r * h_s
        h_m = h * r_s
        h_p = h_m + h
        r_p = r_m + r
        h = self.convh_1(h_p)
        r = self.convr_1(r_p)
        h = self.sigmoid(h)
        r = self.tanh(r)
        # f, _ = self.gate(f.unsqueeze(0))  # 使用门机制选取互补特征
        final = r * h
        return final  # 32 channel




class IGRM(nn.Module):
    def __init__(self):
        super(IGRM, self).__init__()

        self.pfm = PFM_1()
        
        # Multi-Head Self-Attention replaces LSTM for better efficiency and parallel processing
        self.attention = MultiHeadSpatialAttention(in_channels=32, num_heads=4)

        self.conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.baseBlock = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, r, h):
        p = self.pfm(r, h)
        bb_1 = self.baseBlock(p)
        
        # Multi-Head Self-Attention (replaces LSTM)
        output_up = self.attention(bb_1)

        bb_2 = self.baseBlock(output_up)
        con = self.conv(bb_2)
        final = con + r

        return final
