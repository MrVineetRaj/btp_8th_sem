import torch
import torch.nn as nn


class WAFM(nn.Module):
    """Wavelet Aware Feature Module - refines wavelet features with residual blocks.
    
    Analogous to EAFM (Edge Aware Feature Module), this module processes wavelet
    subband features through residual convolutional blocks to enhance texture
    representation.
    
    Input: 12 channels (4 subbands × 3 RGB channels from WaveletMap)
    Output: 12 channels (refined wavelet features)
    """
    
    def __init__(self, in_channels=12, hidden_channels=48):
        super(WAFM, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Input projection: 12 -> 48 channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Residual blocks for feature refinement
        self.baseblock = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Output projection: 48 -> 12 channels
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, w):
        """Refine wavelet features through residual blocks.
        
        Args:
            w: Wavelet features of shape (B, 12, H, W)
            
        Returns:
            Refined wavelet features of shape (B, 12, H, W)
        """
        # Input projection
        x = self.conv1(w)
        
        # First residual block
        out1 = x + self.baseblock(x)
        
        # Second residual block
        out2 = out1 + self.baseblock(out1)
        
        # Output projection
        out3 = self.conv2(out2)
        
        # Global residual connection
        out3 = out3 + w
        
        return out3


class WaveletFeatureFusion(nn.Module):
    """Fuses edge features (8ch) and wavelet features (12ch) into unified representation.
    
    Uses attention-based fusion to adaptively weight edge and wavelet contributions
    based on local image content.
    """
    
    def __init__(self, edge_channels=8, wavelet_channels=12, out_channels=8):
        super(WaveletFeatureFusion, self).__init__()
        
        self.edge_channels = edge_channels
        self.wavelet_channels = wavelet_channels
        self.out_channels = out_channels
        total_channels = edge_channels + wavelet_channels
        
        # Channel attention for adaptive fusion
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 2, total_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, edge_features, wavelet_features):
        """Fuse edge and wavelet features.
        
        Args:
            edge_features: Shape (B, 8, H, W)
            wavelet_features: Shape (B, 12, H, W)
            
        Returns:
            Fused features of shape (B, out_channels, H, W)
        """
        # Concatenate features
        combined = torch.cat([edge_features, wavelet_features], dim=1)
        
        # Apply channel attention
        attention = self.channel_attention(combined)
        attended = combined * attention
        
        # Spatial fusion
        fused = self.fusion_conv(attended)
        
        return fused
