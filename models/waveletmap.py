import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletMap(nn.Module):
    """Extracts wavelet subbands (LL, LH, HL, HH) using Haar wavelet transform.
    
    The Discrete Wavelet Transform decomposes the image into different frequency bands:
    - LL (approximation): Low-frequency content, captures overall structure
    - LH (horizontal detail): Horizontal edges/textures
    - HL (vertical detail): Vertical edges/textures  
    - HH (diagonal detail): Diagonal edges/textures
    
    Output: 12-channel tensor (4 subbands × 3 RGB channels)
    """
    
    def __init__(self, num_channels=3):
        super(WaveletMap, self).__init__()
        
        self.num_channels = num_channels
        
        # Haar wavelet filters (normalized)
        # Low-pass filter: [1, 1] / sqrt(2)
        # High-pass filter: [1, -1] / sqrt(2)
        sqrt2 = 1.0 / (2.0 ** 0.5)
        
        # Create 2D separable filters for DWT
        # LL: low-pass in both directions
        ll_filter = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        # LH: low-pass horizontal, high-pass vertical
        lh_filter = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        # HL: high-pass horizontal, low-pass vertical
        hl_filter = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        # HH: high-pass in both directions
        hh_filter = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        
        # Register filters as buffers (not trainable, but move with model to GPU)
        # Shape: (out_channels, in_channels/groups, kH, kW)
        # Using groups=num_channels for depthwise convolution
        self.register_buffer('ll_filter', ll_filter.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1))
        self.register_buffer('lh_filter', lh_filter.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1))
        self.register_buffer('hl_filter', hl_filter.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1))
        self.register_buffer('hh_filter', hh_filter.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1))
    
    def forward(self, x):
        """Apply single-level 2D Haar DWT.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C=3 (RGB)
            
        Returns:
            Wavelet features of shape (B, 12, H, W) containing LL, LH, HL, HH subbands
            concatenated along channel dimension
        """
        # Pad input for valid convolution (reflection padding for better boundary handling)
        x_padded = F.pad(x, (0, 1, 0, 1), mode='reflect')
        
        # Apply depthwise convolution with stride=1 to maintain spatial resolution
        # Each filter produces num_channels output channels
        ll = F.conv2d(x_padded, self.ll_filter, stride=1, groups=self.num_channels)
        lh = F.conv2d(x_padded, self.lh_filter, stride=1, groups=self.num_channels)
        hl = F.conv2d(x_padded, self.hl_filter, stride=1, groups=self.num_channels)
        hh = F.conv2d(x_padded, self.hh_filter, stride=1, groups=self.num_channels)
        
        # Concatenate all subbands: (B, 12, H, W)
        wavelet_features = torch.cat([ll, lh, hl, hh], dim=1)
        
        return wavelet_features


class MultiScaleWaveletMap(nn.Module):
    """Multi-scale wavelet decomposition for richer texture representation.
    
    Performs 2-level DWT decomposition:
    - Level 1: LL1, LH1, HL1, HH1 (12 channels at original resolution)
    - Level 2: LL2, LH2, HL2, HH2 from LL1 (12 channels, upsampled)
    
    Total output: 24 channels (or 12 if only using detail subbands from level 2)
    """
    
    def __init__(self, num_channels=3, num_levels=2):
        super(MultiScaleWaveletMap, self).__init__()
        
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.wavelet = WaveletMap(num_channels)
    
    def forward(self, x):
        """Apply multi-level DWT.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Multi-scale wavelet features (B, 12*num_levels, H, W)
        """
        B, C, H, W = x.shape
        all_features = []
        
        current = x
        for level in range(self.num_levels):
            # Apply DWT at current scale
            wavelet_out = self.wavelet(current)  # (B, 12, H', W')
            
            # Upsample to original resolution if needed
            if wavelet_out.shape[2:] != (H, W):
                wavelet_out = F.interpolate(wavelet_out, size=(H, W), mode='bilinear', align_corners=False)
            
            all_features.append(wavelet_out)
            
            # Use LL subband (first 3 channels) for next level, downsampled
            ll = wavelet_out[:, :self.num_channels, :, :]
            current = F.avg_pool2d(ll, kernel_size=2, stride=2)
        
        # Concatenate all levels
        return torch.cat(all_features, dim=1)
