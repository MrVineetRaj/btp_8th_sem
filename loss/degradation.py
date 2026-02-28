"""Degradation-aware loss functions for robust super-resolution.

This module provides loss functions for:
1. Supervising degradation estimation (kernel + noise level)
2. Degradation-aware reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationLoss(nn.Module):
    """Loss for supervising degradation estimation.
    
    This loss compares the estimated blur kernel and noise level
    with ground truth degradation parameters.
    
    Args:
        kernel_weight: Weight for kernel estimation loss (default: 1.0)
        noise_weight: Weight for noise level estimation loss (default: 0.1)
    """
    
    def __init__(self, kernel_weight=1.0, noise_weight=0.1):
        super(DegradationLoss, self).__init__()
        self.kernel_weight = kernel_weight
        self.noise_weight = noise_weight
        self.kernel_loss = nn.L1Loss()
        self.noise_loss = nn.L1Loss()
        
    def forward(self, pred_kernel, pred_noise, gt_kernel, gt_noise):
        """Compute degradation estimation loss.
        
        Args:
            pred_kernel: Predicted blur kernel (B, 1, K, K)
            pred_noise: Predicted noise level (B, 1)
            gt_kernel: Ground truth blur kernel (B, K, K)
            gt_noise: Ground truth noise level (B, 1)
            
        Returns:
            Combined degradation loss
        """
        if gt_kernel.dim() == 3:
            gt_kernel = gt_kernel.unsqueeze(1)
        
        k_loss = self.kernel_loss(pred_kernel, gt_kernel)
        n_loss = self.noise_loss(pred_noise, gt_noise)
        
        total_loss = self.kernel_weight * k_loss + self.noise_weight * n_loss
        
        return total_loss


class KernelLoss(nn.Module):
    """Loss for blur kernel estimation only.
    
    Supports multiple loss types:
    - L1: Mean absolute error
    - L2: Mean squared error
    - KL: KL divergence (treats kernels as probability distributions)
    """
    
    def __init__(self, loss_type='L1'):
        super(KernelLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'L1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'L2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'KL':
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
    def forward(self, pred_kernel, gt_kernel):
        """Compute kernel estimation loss.
        
        Args:
            pred_kernel: Predicted blur kernel (B, 1, K, K)
            gt_kernel: Ground truth blur kernel (B, K, K) or (B, 1, K, K)
            
        Returns:
            Kernel loss value
        """
        if gt_kernel.dim() == 3:
            gt_kernel = gt_kernel.unsqueeze(1)
            
        if self.loss_type == 'KL':
            pred_log = torch.log(pred_kernel + 1e-10)
            return self.loss_fn(pred_log, gt_kernel)
        else:
            return self.loss_fn(pred_kernel, gt_kernel)


class NoiseLevelLoss(nn.Module):
    """Loss for noise level estimation.
    
    Uses relative error to handle varying noise scales.
    """
    
    def __init__(self, use_relative=True):
        super(NoiseLevelLoss, self).__init__()
        self.use_relative = use_relative
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred_noise, gt_noise):
        """Compute noise level estimation loss.
        
        Args:
            pred_noise: Predicted noise level (B, 1)
            gt_noise: Ground truth noise level (B, 1)
            
        Returns:
            Noise level loss value
        """
        if self.use_relative:
            relative_error = torch.abs(pred_noise - gt_noise) / (gt_noise + 1e-6)
            return relative_error.mean()
        else:
            return self.l1_loss(pred_noise, gt_noise)


class DegradationAwareReconstructionLoss(nn.Module):
    """Reconstruction loss weighted by estimated degradation severity.
    
    The idea is to reduce the penalty for difficult (highly degraded) cases,
    allowing the model to focus on learnable patterns.
    
    Args:
        base_loss: Base reconstruction loss ('L1', 'L2', or 'Charbonnier')
        adaptive_weight: Whether to use adaptive weighting based on noise
    """
    
    def __init__(self, base_loss='L1', adaptive_weight=True):
        super(DegradationAwareReconstructionLoss, self).__init__()
        self.adaptive_weight = adaptive_weight
        
        if base_loss == 'L1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif base_loss == 'L2':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif base_loss == 'Charbonnier':
            self.loss_fn = CharbonnierLoss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
            
    def forward(self, sr, hr, noise_level=None):
        """Compute degradation-aware reconstruction loss.
        
        Args:
            sr: Super-resolved image (B, C, H, W)
            hr: High-resolution ground truth (B, C, H, W)
            noise_level: Estimated noise level (B, 1) or None
            
        Returns:
            Weighted reconstruction loss
        """
        loss = self.loss_fn(sr, hr)
        
        if self.adaptive_weight and noise_level is not None:
            weight = 1.0 / (1.0 + noise_level.view(-1, 1, 1, 1) * 0.01)
            weight = torch.clamp(weight, 0.5, 1.5)
            loss = loss * weight
            
        return loss.mean()


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (robust L1 loss).
    
    L(x, y) = sqrt((x - y)^2 + eps^2)
    
    This is a smooth approximation to L1 loss that is differentiable at 0.
    """
    
    def __init__(self, eps=1e-6, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class CombinedDegradationLoss(nn.Module):
    """Combined loss for training with degradation estimation.
    
    Combines:
    1. Reconstruction loss (SR vs HR)
    2. Degradation estimation loss (predicted vs GT params)
    
    Args:
        recon_weight: Weight for reconstruction loss
        deg_weight: Weight for degradation estimation loss
        kernel_weight: Weight for kernel loss within degradation loss
        noise_weight: Weight for noise loss within degradation loss
    """
    
    def __init__(self, recon_weight=1.0, deg_weight=0.1, 
                 kernel_weight=1.0, noise_weight=0.1):
        super(CombinedDegradationLoss, self).__init__()
        self.recon_weight = recon_weight
        self.deg_weight = deg_weight
        
        self.recon_loss = nn.L1Loss()
        self.deg_loss = DegradationLoss(kernel_weight, noise_weight)
        
    def forward(self, sr, hr, pred_kernel=None, pred_noise=None,
                gt_kernel=None, gt_noise=None):
        """Compute combined loss.
        
        Args:
            sr: Super-resolved image
            hr: High-resolution ground truth
            pred_kernel: Predicted blur kernel (optional)
            pred_noise: Predicted noise level (optional)
            gt_kernel: Ground truth blur kernel (optional)
            gt_noise: Ground truth noise level (optional)
            
        Returns:
            Total combined loss
        """
        recon = self.recon_loss(sr, hr)
        total = self.recon_weight * recon
        
        if all(v is not None for v in [pred_kernel, pred_noise, gt_kernel, gt_noise]):
            deg = self.deg_loss(pred_kernel, pred_noise, gt_kernel, gt_noise)
            total = total + self.deg_weight * deg
            
        return total
