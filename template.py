"""Template configurations for different model presets.

This module allows defining preset configurations for various models
and training scenarios. Use --template argument to select a preset.
"""


def set_template(args):
    """Set predefined argument templates.
    
    Args:
        args: Argument namespace from option.py
        
    Available templates:
        - '.': No template (default)
        - 'EPGDUN': Standard EPGDUN configuration
        - 'EPGDUN_DEG': EPGDUN with degradation-aware training
    """
    if args.template == '.':
        return
    
    if args.template == 'EPGDUN':
        args.model = 'EPGDUN'
        args.scale = '2'
        args.patch_size = 96
        args.batch_size = 4
        args.epochs = 300
        args.lr = 1e-4
        args.loss = '1*L1'
    
    elif args.template == 'EPGDUN_DEG':
        args.model = 'EPGDUN'
        args.scale = '2'
        args.patch_size = 96
        args.batch_size = 4
        args.epochs = 300
        args.lr = 1e-4
        args.use_degradation = True
        args.loss = '1*L1+0.1*DEG'
        args.blur_sigma_range = '0.2,3.0'
        args.jpeg_quality_range = '30,95'
        args.degradation_prob = 0.5
    
    elif args.template == 'EPGDUN_X4':
        args.model = 'EPGDUN'
        args.scale = '4'
        args.patch_size = 192
        args.batch_size = 2
        args.epochs = 300
        args.lr = 1e-4
        args.loss = '1*L1'
