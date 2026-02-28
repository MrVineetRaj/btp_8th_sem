"""Noisy benchmark dataset for evaluating robustness to real-world degradations.

This module provides a benchmark dataset class that applies various degradations
to test images, enabling evaluation of model robustness.
"""

import os
import random
import numpy as np
import imageio

from mydata import srdata
import mydata.common as common


class BenchmarkNoise(srdata.SRData):
    """Benchmark dataset with added noise and degradations.
    
    This class extends the standard benchmark to add various degradations
    during testing, allowing evaluation of model robustness to:
    - Gaussian noise
    - Blur (Gaussian and motion)
    - JPEG compression artifacts
    
    Args:
        args: Argument namespace containing:
            - noise_level: Gaussian noise sigma (default: 15)
            - blur_sigma: Gaussian blur sigma (default: 1.0)
            - jpeg_quality: JPEG quality (default: 70)
            - degradation_type: Type of degradation ('noise', 'blur', 'jpeg', 'all')
        train: Whether this is for training (should be False for benchmark)
    """
    
    def __init__(self, args, train=False):
        super(BenchmarkNoise, self).__init__(args, train, benchmark=True)
        
        self.noise_level = getattr(args, 'noise_level', 15)
        self.blur_sigma = getattr(args, 'blur_sigma', 1.0)
        self.jpeg_quality = getattr(args, 'jpeg_quality', 70)
        self.degradation_type = getattr(args, 'degradation_type', 'noise')
        
    def _scan(self):
        """Scan directory for HR and LR image pairs."""
        list_hr = []
        list_lr = [[] for _ in self.scale]
        
        hr_path = os.path.join(self.dir_hr, 'X{}'.format(self.scale[0]))
        if not os.path.exists(hr_path):
            hr_path = self.dir_hr
            
        for entry in os.scandir(hr_path):
            filename = os.path.splitext(entry.name)[0]
            list_hr.append(os.path.join(hr_path, filename + self.ext))
            for si, s in enumerate(self.scale):
                lr_path = os.path.join(
                    self.dir_lr,
                    'X{}/{}{}'.format(s, filename, self.ext)
                )
                if not os.path.exists(lr_path):
                    lr_path = os.path.join(self.dir_lr, filename + self.ext)
                list_lr[si].append(lr_path)
        
        list_hr.sort()
        for l in list_lr:
            l.sort()
        
        return list_hr, list_lr
    
    def _set_filesystem(self, dir_data):
        """Set up file system paths for the dataset."""
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = '.png'
    
    def _load_file(self, idx):
        """Load and apply degradation to the LR image."""
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = imageio.imread(lr)
            hr = imageio.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)
        
        lr = self._apply_degradation(lr)
        
        filename = os.path.splitext(os.path.split(filename)[-1])[0]
        
        return lr, hr, filename
    
    def _apply_degradation(self, image):
        """Apply specified degradation to the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Degraded image
        """
        degraded = image.copy()
        deg_type = self.degradation_type.lower()
        
        if deg_type == 'noise' or deg_type == 'all':
            degraded = common.add_noise(degraded, str(self.noise_level))
        
        if deg_type == 'blur' or deg_type == 'all':
            degraded = common.apply_gaussian_blur(degraded, self.blur_sigma)
        
        if deg_type == 'jpeg' or deg_type == 'all':
            degraded = common.apply_jpeg_compression(degraded, self.jpeg_quality)
        
        if deg_type == 'motion':
            angle = random.uniform(0, 180)
            length = random.uniform(5, 15)
            degraded = common.apply_motion_blur(degraded, angle, length)
        
        return degraded


class BenchmarkNoiseMulti(BenchmarkNoise):
    """Benchmark with multiple degradation levels for comprehensive evaluation.
    
    This class creates multiple versions of each test image with different
    degradation levels, useful for plotting degradation vs. performance curves.
    """
    
    def __init__(self, args, train=False):
        super(BenchmarkNoiseMulti, self).__init__(args, train)
        
        self.noise_levels = [0, 5, 10, 15, 20, 25, 30]
        self.blur_sigmas = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        self.jpeg_qualities = [100, 90, 80, 70, 60, 50, 40]
        
        self._current_level_idx = 0
        
    def set_degradation_level(self, level_idx):
        """Set the current degradation level index.
        
        Args:
            level_idx: Index into the degradation level arrays
        """
        self._current_level_idx = level_idx
        self.noise_level = self.noise_levels[level_idx]
        self.blur_sigma = self.blur_sigmas[level_idx]
        self.jpeg_quality = self.jpeg_qualities[level_idx]
    
    def get_num_levels(self):
        """Return the number of degradation levels."""
        return len(self.noise_levels)


class BenchmarkReal(BenchmarkNoise):
    """Benchmark for real-world degraded images.
    
    This class assumes that degraded images are stored in a separate folder
    rather than being synthetically generated. Useful for testing on
    real-world datasets like RealSR.
    """
    
    def __init__(self, args, train=False):
        super(BenchmarkReal, self).__init__(args, train)
        
    def _set_filesystem(self, dir_data):
        """Set up file system paths for real-world degraded images."""
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_real')
        self.ext = '.png'
    
    def _load_file(self, idx):
        """Load files without applying synthetic degradation."""
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = imageio.imread(lr)
            hr = imageio.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)
        
        filename = os.path.splitext(os.path.split(filename)[-1])[0]
        
        return lr, hr, filename
