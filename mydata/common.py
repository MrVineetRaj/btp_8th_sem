"""Common utility functions for data processing and degradation simulation."""

import random
import numpy as np
import torch
import cv2
from io import BytesIO
from PIL import Image


def set_channel(images, n_colors):
    """Set the number of color channels for images.
    
    Args:
        images: List of numpy arrays (H, W) or (H, W, C)
        n_colors: Target number of channels (1 for grayscale, 3 for RGB)
    
    Returns:
        List of images with the specified number of channels
    """
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        
        c = img.shape[2]
        if n_colors == 1 and c == 3:
            img = np.expand_dims(img[:, :, 0], axis=2)
        elif n_colors == 3 and c == 1:
            img = np.concatenate([img] * 3, axis=2)
        
        return img
    
    return [_set_channel(img) for img in images]


def np2Tensor(images, rgb_range):
    """Convert numpy arrays to PyTorch tensors.
    
    Args:
        images: List of numpy arrays (H, W, C) in range [0, 255]
        rgb_range: Target range for pixel values
    
    Returns:
        List of tensors (C, H, W) in range [0, rgb_range]
    """
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.0)
        return tensor
    
    return [_np2Tensor(img) for img in images]


def get_patch(lr, hr, patch_size, scale, multi_scale=False):
    """Extract random patches from LR and HR image pairs.
    
    Args:
        lr: Low-resolution image (H, W, C)
        hr: High-resolution image (H, W, C)
        patch_size: Size of the HR patch
        scale: Super-resolution scale factor
        multi_scale: Whether to use multi-scale training
    
    Returns:
        Tuple of (lr_patch, hr_patch)
    """
    ih, iw = lr.shape[:2]
    
    if multi_scale:
        p = scale
    else:
        p = 1
    
    tp = p * patch_size
    ip = tp // scale
    
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    
    tx, ty = scale * ix, scale * iy
    
    lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
    hr_patch = hr[ty:ty + tp, tx:tx + tp, :]
    
    return lr_patch, hr_patch


def augment(images, hflip=True, rot=True):
    """Apply random geometric augmentations.
    
    Args:
        images: List of numpy arrays
        hflip: Enable horizontal flip
        rot: Enable rotation
    
    Returns:
        List of augmented images
    """
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    
    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return np.ascontiguousarray(img)
    
    return [_augment(img) for img in images]


def add_noise(image, noise_sigma):
    """Add Gaussian noise to image.
    
    Args:
        image: Numpy array (H, W, C) in range [0, 255]
        noise_sigma: Noise standard deviation or '.' for no noise
    
    Returns:
        Noisy image
    """
    if noise_sigma == '.' or noise_sigma is None:
        return image
    
    try:
        sigma = float(noise_sigma)
    except (ValueError, TypeError):
        return image
    
    if sigma <= 0:
        return image
    
    noise = np.random.randn(*image.shape) * sigma
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def generate_gaussian_kernel(kernel_size, sigma):
    """Generate a 2D Gaussian blur kernel.
    
    Args:
        kernel_size: Size of the kernel (must be odd)
        sigma: Standard deviation of the Gaussian
    
    Returns:
        Normalized Gaussian kernel as numpy array
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    x = np.arange(kernel_size) - kernel_size // 2
    gauss_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel = np.outer(gauss_1d, gauss_1d)
    kernel = kernel / kernel.sum()
    
    return kernel.astype(np.float32)


def generate_motion_blur_kernel(kernel_size, angle, length):
    """Generate a motion blur kernel.
    
    Args:
        kernel_size: Size of the kernel
        angle: Angle of motion in degrees
        length: Length of motion blur
    
    Returns:
        Normalized motion blur kernel
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    
    angle_rad = np.deg2rad(angle)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    for i in range(int(length)):
        x = int(center + i * dx - length * dx / 2)
        y = int(center + i * dy - length * dy / 2)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    if kernel.sum() == 0:
        kernel[center, center] = 1
    
    kernel = kernel / kernel.sum()
    return kernel


def apply_gaussian_blur(image, sigma):
    """Apply Gaussian blur to image.
    
    Args:
        image: Numpy array (H, W, C)
        sigma: Blur sigma
    
    Returns:
        Blurred image
    """
    if sigma <= 0:
        return image
    
    kernel_size = int(np.ceil(sigma * 6))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, 21))
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred


def apply_motion_blur(image, angle, length):
    """Apply motion blur to image.
    
    Args:
        image: Numpy array (H, W, C)
        angle: Angle of motion in degrees
        length: Length of motion blur
    
    Returns:
        Motion blurred image
    """
    if length <= 1:
        return image
    
    kernel_size = int(length) + 2
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, 31))
    
    kernel = generate_motion_blur_kernel(kernel_size, angle, length)
    blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred


def apply_jpeg_compression(image, quality):
    """Apply JPEG compression artifacts.
    
    Args:
        image: Numpy array (H, W, C) in RGB format
        quality: JPEG quality (1-100, lower = more artifacts)
    
    Returns:
        Compressed image
    """
    quality = max(1, min(100, int(quality)))
    
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed_array = np.array(compressed)
    
    return compressed_array


def apply_downsampling(image, scale, method='bicubic'):
    """Apply downsampling with various methods.
    
    Args:
        image: Numpy array (H, W, C)
        scale: Downsampling scale factor
        method: 'bicubic', 'bilinear', 'area', or 'nearest'
    
    Returns:
        Downsampled image
    """
    h, w = image.shape[:2]
    new_h, new_w = h // scale, w // scale
    
    interpolation_methods = {
        'bicubic': cv2.INTER_CUBIC,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'nearest': cv2.INTER_NEAREST
    }
    
    interp = interpolation_methods.get(method, cv2.INTER_CUBIC)
    downsampled = cv2.resize(image, (new_w, new_h), interpolation=interp)
    
    return downsampled


def add_poisson_noise(image, scale=1.0):
    """Add Poisson noise (shot noise) to image.
    
    Args:
        image: Numpy array (H, W, C) in range [0, 255]
        scale: Noise scale factor
    
    Returns:
        Noisy image
    """
    if scale <= 0:
        return image
    
    image_float = image.astype(np.float32) / 255.0
    noisy = np.random.poisson(image_float * 255 * scale) / (255 * scale)
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    
    return noisy


def add_degradation(image, args):
    """Apply realistic degradation pipeline to image.
    
    This function applies a combination of blur, noise, and compression
    to simulate real-world image degradation.
    
    Args:
        image: Numpy array (H, W, C) in range [0, 255]
        args: Argument namespace containing degradation parameters:
            - blur_sigma_range: str "min,max" for Gaussian blur sigma
            - jpeg_quality_range: str "min,max" for JPEG quality
            - degradation_prob: float, probability of applying each degradation
    
    Returns:
        Tuple of (degraded_image, degradation_params)
        degradation_params is a dict containing the applied degradation parameters
    """
    deg_params = {
        'blur_type': 'none',
        'blur_sigma': 0.0,
        'blur_kernel': None,
        'motion_angle': 0.0,
        'motion_length': 0.0,
        'noise_sigma': 0.0,
        'jpeg_quality': 100,
        'poisson_scale': 0.0
    }
    
    degraded = image.copy()
    prob = getattr(args, 'degradation_prob', 0.5)
    
    blur_sigma_range = getattr(args, 'blur_sigma_range', '0.2,3.0')
    try:
        blur_min, blur_max = map(float, blur_sigma_range.split(','))
    except:
        blur_min, blur_max = 0.2, 3.0
    
    jpeg_quality_range = getattr(args, 'jpeg_quality_range', '30,95')
    try:
        jpeg_min, jpeg_max = map(int, jpeg_quality_range.split(','))
    except:
        jpeg_min, jpeg_max = 30, 95
    
    if random.random() < prob:
        blur_type = random.choice(['gaussian', 'motion'])
        
        if blur_type == 'gaussian':
            sigma = random.uniform(blur_min, blur_max)
            degraded = apply_gaussian_blur(degraded, sigma)
            deg_params['blur_type'] = 'gaussian'
            deg_params['blur_sigma'] = sigma
            
            kernel_size = int(np.ceil(sigma * 6))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, min(kernel_size, 21))
            deg_params['blur_kernel'] = generate_gaussian_kernel(kernel_size, sigma)
        else:
            angle = random.uniform(0, 180)
            length = random.uniform(3, 15)
            degraded = apply_motion_blur(degraded, angle, length)
            deg_params['blur_type'] = 'motion'
            deg_params['motion_angle'] = angle
            deg_params['motion_length'] = length
            
            kernel_size = int(length) + 2
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, min(kernel_size, 31))
            deg_params['blur_kernel'] = generate_motion_blur_kernel(kernel_size, angle, length)
    
    if random.random() < prob:
        noise_sigma = random.uniform(1, 25)
        degraded = add_noise(degraded, str(noise_sigma))
        deg_params['noise_sigma'] = noise_sigma
    
    if random.random() < prob * 0.5:
        poisson_scale = random.uniform(0.5, 2.0)
        degraded = add_poisson_noise(degraded, poisson_scale)
        deg_params['poisson_scale'] = poisson_scale
    
    if random.random() < prob:
        quality = random.randint(jpeg_min, jpeg_max)
        degraded = apply_jpeg_compression(degraded, quality)
        deg_params['jpeg_quality'] = quality
    
    return degraded, deg_params


def get_degradation_kernel_tensor(deg_params, kernel_size=21):
    """Convert degradation parameters to a tensor kernel for model supervision.
    
    Args:
        deg_params: Dict containing degradation parameters
        kernel_size: Target kernel size
    
    Returns:
        Tensor of shape (kernel_size, kernel_size) representing the blur kernel
    """
    kernel = deg_params.get('blur_kernel', None)
    
    if kernel is None:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, kernel_size // 2] = 1.0
    else:
        current_size = kernel.shape[0]
        if current_size != kernel_size:
            pad_total = kernel_size - current_size
            if pad_total > 0:
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                kernel = np.pad(kernel, ((pad_before, pad_after), (pad_before, pad_after)))
            else:
                start = (current_size - kernel_size) // 2
                kernel = kernel[start:start+kernel_size, start:start+kernel_size]
    
    return torch.from_numpy(kernel).float()


def get_noise_level_tensor(deg_params):
    """Extract noise level from degradation parameters as tensor.
    
    Args:
        deg_params: Dict containing degradation parameters
    
    Returns:
        Tensor of shape (1,) containing the noise sigma
    """
    noise_sigma = deg_params.get('noise_sigma', 0.0)
    poisson_scale = deg_params.get('poisson_scale', 0.0)
    
    total_noise = np.sqrt(noise_sigma**2 + (poisson_scale * 10)**2)
    
    return torch.tensor([total_noise]).float()
