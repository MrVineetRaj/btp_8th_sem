import os
from importlib import import_module


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


matplotlib.use('Agg')   # Render images to PNG/PDF format without display


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()

        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []  # List storing dicts {type, weight, function}
        self.loss_module = nn.ModuleList()
        self.log = torch.Tensor()  # Loss log for storing training and validation losses
        device = torch.device('cpu' if args.cpu else 'cuda')

        # Degradation loss settings
        self.use_degradation = getattr(args, 'use_degradation', False)
        self.deg_loss_weight = getattr(args, 'deg_loss_weight', 0.1)
        self.degradation_loss = None
        
        # Step 1: Prepare loss function list
        for loss in args.loss.split('+'):
            # Get weight, loss_type
            weight, loss_type = loss.split('*')  # Input format: weight*loss_type
            # MSE
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            # L1
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            # VGG
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],  # Remove 'VGG' prefix
                    rgb_range=args.rgb_range    # Image pixel intensity range
                )
            # GAN
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')  # Adversarial is an abstract base class
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            # DEG - Degradation estimation loss
            elif loss_type == 'DEG':
                module = import_module('loss.degradation')
                loss_function = getattr(module, 'DegradationLoss')(
                    kernel_weight=1.0,
                    noise_weight=0.1
                )
                self.degradation_loss = loss_function
            # Charbonnier loss
            elif loss_type == 'Charbonnier':
                module = import_module('loss.degradation')
                loss_function = getattr(module, 'CharbonnierLoss')()
            # Add to loss list
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })
            if loss_type.find('GAN') >= 0:  # GAN also needs discriminator loss
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})
        # If more than one loss function, add total loss to track combined loss
        if len(self.loss) > 1:  # Add total loss function to record combined loss
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
            # type=Total, weight=weight of this loss in total loss
        # Step 2: Load loss functions into loss_module
        for l in self.loss:
            if l['function'] is not None:   # All implemented loss functions
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])  # Add to loss module for computing total loss

        # Step 3: Move model to device
        self.loss_module.to(device)

        # Step 4: Configure model precision
        if args.precision == 'half':
            self.loss_module.half()

        # Step 5: If not using CPU and GPUs > 1, use DataParallel for multi-GPU training
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )
        # Step 6: Load existing config if available
        if args.load != '.':    # . means current directory, otherwise load pretrained model
            self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr, deg_params=None):
        """Compute weighted multi-loss.
        :param sr: Super-resolved image
        :param hr: Original high-resolution image
        :param deg_params: Optional degradation params dict containing:
            - 'pred_kernel': Predicted blur kernel
            - 'pred_noise': Predicted noise level
            - 'gt_kernel': Ground truth blur kernel
            - 'gt_noise': Ground truth noise level
        :return: Combined weighted loss
        """
        losses = []
        # Iterate all loss functions and record effective_loss to losses list
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                # Handle degradation loss specially
                if l['type'] == 'DEG':
                    if deg_params is not None and all(k in deg_params for k in ['pred_kernel', 'pred_noise', 'gt_kernel', 'gt_noise']):
                        loss = l['function'](
                            deg_params['pred_kernel'],
                            deg_params['pred_noise'],
                            deg_params['gt_kernel'],
                            deg_params['gt_noise']
                        )
                    else:
                        continue  # Skip DEG loss if params not available
                else:
                    loss = l['function'](sr, hr)  # Compute loss with sr and hr
                effective_loss = l['weight'] * loss  # effective_loss = weight * loss
                losses.append(effective_loss)
                # item converts zero-dim tensor to float, record effective_loss to log
                self.log[-1, i] += effective_loss.item()    # Last row, column i
            elif l['type'] == 'DIS':
                # Total loss is defined after discriminator loss, so use i-1 to get discriminator loss
                self.log[-1, i] += self.loss[i - 1]['function'].loss
        # Sum losses from different loss functions
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):  # If loss module has learning rate scheduler
                l.scheduler.step()

    def start_log(self):
        """Start loss log recording, each record has shape=(1, num_loss_functions)"""
        self.log = torch.cat(
            (self.log, torch.zeros(1, len(self.loss)))
        )

    def end_log(self, n_batches):
        """End loss log recording"""
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        """Display current loss status, divided by batch.
        :return: str
        """
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        """Plot loss-epoch curves for each function, save as pdf.
        :param apath: File storage path
        :param epoch: epoch
        """
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):  # Iterate loss function list
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def get_loss_module(self):
        """Return loss function.
        :return: loss_module
        """
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        """Save model state and loss log"""
        torch.save(self.state_dict(),  # Save model state
                   os.path.join(apath, 'loss.pt'))
        torch.save(self.log,  # Save loss log
                   os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:  # If loading to CPU, need extra settings
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        # Load pretrained weights into model using load_state_dict
        self.load_state_dict(
            # torch.load loads objects saved by torch.save
            torch.load(os.path.join(apath, 'loss.pt'), **kwargs)
        )
        # Load loss log
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        # Update learning rate based on log length
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)):
                    l.scheduler.step()  # Learning rate update
