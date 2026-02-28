"""Degradation Estimation Module for Real-World Super-Resolution.

This module estimates blur kernels and noise levels from low-resolution images,
enabling the model to adapt to various real-world degradation types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block for feature extraction."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)


class DegradationEstimator(nn.Module):
    """Estimates blur kernel and noise level from LR images.
    
    This lightweight network analyzes the LR input to estimate:
    1. A blur kernel (spatially-variant or fixed size)
    2. A noise level (standard deviation)
    
    These estimates are used to guide the super-resolution process
    and make it robust to real-world degradations.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        kernel_size: Size of the estimated blur kernel (default: 21)
        num_features: Number of intermediate features (default: 64)
    """
    
    def __init__(self, in_channels=3, kernel_size=21, num_features=64):
        super(DegradationEstimator, self).__init__()
        
        self.kernel_size = kernel_size
        self.num_features = num_features
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features * 2),
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features * 4),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.kernel_head = nn.Sequential(
            nn.Linear(num_features * 4, num_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_features * 2, kernel_size * kernel_size),
        )
        
        self.noise_head = nn.Sequential(
            nn.Linear(num_features * 4, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, 1),
            nn.Softplus(),
        )
        
    def forward(self, x):
        """Estimate degradation parameters from input.
        
        Args:
            x: Input LR image tensor of shape (B, C, H, W)
            
        Returns:
            blur_kernel: Estimated blur kernel of shape (B, 1, kernel_size, kernel_size)
            noise_level: Estimated noise level of shape (B, 1)
        """
        batch_size = x.size(0)
        
        features = self.encoder(x)
        pooled = self.global_pool(features).view(batch_size, -1)
        
        kernel_flat = self.kernel_head(pooled)
        kernel = kernel_flat.view(batch_size, 1, self.kernel_size, self.kernel_size)
        kernel = F.softmax(kernel.view(batch_size, -1), dim=1)
        kernel = kernel.view(batch_size, 1, self.kernel_size, self.kernel_size)
        
        noise_level = self.noise_head(pooled)
        
        return kernel, noise_level


class DegradationAwareConv(nn.Module):
    """Convolution that uses estimated blur kernel for degradation modeling.
    
    This module applies the estimated blur kernel to approximate the
    degradation process, useful in the data fidelity term.
    
    Args:
        kernel_size: Size of the blur kernel
    """
    
    def __init__(self, kernel_size=21):
        super(DegradationAwareConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
    def forward(self, x, kernel):
        """Apply estimated blur kernel to input.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            kernel: Blur kernel of shape (B, 1, K, K)
            
        Returns:
            Blurred output of shape (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()
        
        kernel = kernel.expand(batch_size, channels, self.kernel_size, self.kernel_size)
        
        output = []
        for b in range(batch_size):
            k = kernel[b:b+1]
            k = k.view(channels, 1, self.kernel_size, self.kernel_size)
            
            blurred = F.conv2d(
                x[b:b+1], 
                k, 
                padding=self.padding, 
                groups=channels
            )
            output.append(blurred)
        
        return torch.cat(output, dim=0)


class LightweightDegradationEstimator(nn.Module):
    """A more lightweight version of DegradationEstimator for efficiency.
    
    Uses fewer parameters while still providing reasonable estimates.
    Suitable for mobile/edge deployment scenarios.
    
    Args:
        in_channels: Number of input channels
        kernel_size: Size of the estimated blur kernel
    """
    
    def __init__(self, in_channels=3, kernel_size=21):
        super(LightweightDegradationEstimator, self).__init__()
        
        self.kernel_size = kernel_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.kernel_head = nn.Linear(128, kernel_size * kernel_size)
        self.noise_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        features = self.encoder(x)
        pooled = self.global_pool(features).view(batch_size, -1)
        
        kernel_flat = self.kernel_head(pooled)
        kernel = F.softmax(kernel_flat, dim=1)
        kernel = kernel.view(batch_size, 1, self.kernel_size, self.kernel_size)
        
        noise_level = self.noise_head(pooled)
        
        return kernel, noise_level


def create_identity_kernel(kernel_size, batch_size=1, device='cpu'):
    """Create an identity (delta) kernel.
    
    Args:
        kernel_size: Size of the kernel
        batch_size: Batch size
        device: Device to create tensor on
        
    Returns:
        Identity kernel of shape (batch_size, 1, kernel_size, kernel_size)
    """
    kernel = torch.zeros(batch_size, 1, kernel_size, kernel_size, device=device)
    center = kernel_size // 2
    kernel[:, :, center, center] = 1.0
    return kernel
