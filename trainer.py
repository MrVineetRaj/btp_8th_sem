import os
from decimal import Decimal

import torch
from tqdm import tqdm

import utility
import time
import torch.nn.functional as F

class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale  # Scale factor

        self.ckp = ckp  # Checkpoint

        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        # Model attribute stores the EGDUN object instantiated by make_model
        self.model = my_model
        # Loss attribute stores the loss function instance
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)  # Optimizer
        self.scheduler = utility.make_scheduler(args, self.optimizer)  # Scheduler
        
        # Degradation-aware training settings
        self.use_degradation = getattr(args, 'use_degradation', False)

        if self.args.load != '.':  # Not equal means need to load optimizer state dict
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(ckp.dir, 'optimizer.pt')
                )
            )
            for _ in range(len(ckp.log)):  # Adjust learning rate based on log steps
                self.scheduler.step()

        self.error_last = 1e8

    def test(self):
        epoch = self.scheduler.last_epoch + 2
        self.ckp.write_log('Evaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        # Built-in method of nn.Module, switches model to evaluation mode
        self.model.eval()
        # self.args.test_only = True
        # print("self.args.test_only:", self.args.test_only)  #True
        # Disable autograd for testing, 'with' creates a no-grad context
        # Autograd is re-enabled after exiting the 'with' block
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)  # Progress bar
                for idx_img, batch_data in enumerate(tqdm_test):
                    # Handle both old and new data format
                    if len(batch_data) >= 5:
                        lr, hr, filename = batch_data[0], batch_data[1], batch_data[2]
                    else:
                        lr, hr, filename, _ = batch_data
                    filename = filename[0]  # Test dataset name
                    no_eval = (hr.nelement() == 1)  # Check if HR image has only 1 pixel
                    if not no_eval:  # If HR image has more than 1 pixel
                        lr, hr = self.prepare([lr, hr])  # Preprocess both LR and HR
                    else:  # Otherwise only process LR image
                        lr = self.prepare([lr])[0]
                    sr = self.model(lr, idx_scale)  # Test model
                    if isinstance(sr, list):
                        sr = sr[-1]  # Save last version of reconstructed SR image
                    # print("hr shape:", hr.shape)
                    # print("sr shape:", sr.shape)
                    sr = utility.quantize(sr, self.args.rgb_range)  # Convert tensor to RGB range
                    # hr_size = hr.shape[2:]  # Can try both hr or sr sampling
                    # sr = F.interpolate(sr, size=hr_size, mode='bilinear', align_corners=False)
                    sr_size = sr.shape[2:]
                    hr = F.interpolate(hr, size=sr_size, mode='bilinear', align_corners=False)
                    save_list = [sr]  # Save processed SR image
                    if not no_eval:  # Calculate PSNR
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])
                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)
                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)  # Return max PSNR and its index
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1  # Index starts from 0, +1 to start from 1
                    )
                )
        # print("self.args.test_only_test:", self.args.test_only)

        if not self.args.test_only:  # If not test only, save the model
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))  # Compare if best epoch matches current

    def train(self):
        # torch.cuda.synchronize()
        # start = time.time()
        self.scheduler.step()  # Adjust learning rate
        self.loss.step()  # Optimize loss function
        epoch = self.scheduler.last_epoch + 2
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '\n[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        # Built-in method of nn.Module, switches model to training mode
        self.model.train()
        # self.args.test_only = False
        # print("self.args.test_only_train:", self.args.test_only)
        timer_data, timer_model = utility.timer(), utility.timer()
        # tqdm_train = tqdm(self.loader_train, ncols=80)
        for batch, batch_data in enumerate(self.loader_train):
            # Unpack batch data - handle both old and new format
            if len(batch_data) == 5:
                lr, hr, _, gt_kernel, gt_noise = batch_data
                idx_scale = 0
            elif len(batch_data) == 6:
                lr, hr, _, gt_kernel, gt_noise, idx_scale = batch_data
            else:
                lr, hr, _, idx_scale = batch_data
                gt_kernel, gt_noise = None, None

            lr, hr = self.prepare([lr, hr])  # Convert input LR, HR images to half precision
            
            # Prepare degradation ground truth if available
            if gt_kernel is not None and gt_kernel[0] is not None:
                gt_kernel = self.prepare([gt_kernel])[0]
            else:
                gt_kernel = None
            if gt_noise is not None and gt_noise[0] is not None:
                gt_noise = self.prepare([gt_noise])[0]
            else:
                gt_noise = None

            timer_data.hold()  # Pause timer
            timer_model.tic()  # Restart timer

            self.optimizer.zero_grad()  # Clear gradients
            
            # Forward pass with optional degradation estimation
            if self.use_degradation:
                result = self.model(lr, idx_scale, return_degradation=True)
                if isinstance(result, tuple):
                    sr, deg_pred = result
                    pred_kernel = deg_pred.get('blur_kernel', None)
                    pred_noise = deg_pred.get('noise_level', None)
                else:
                    sr = result
                    pred_kernel, pred_noise = None, None
            else:
                sr = self.model(lr, idx_scale)
                pred_kernel, pred_noise = None, None

            # Prepare degradation parameters dict for loss
            deg_params = None
            if self.use_degradation and pred_kernel is not None:
                deg_params = {
                    'pred_kernel': pred_kernel,
                    'pred_noise': pred_noise,
                    'gt_kernel': gt_kernel,
                    'gt_noise': gt_noise
                }

            # Calculate loss
            if isinstance(sr, list):  # If SR is a list, compute loss for each and average
                loss = 0
                for sr_ in sr:
                    loss += self.loss(sr_, hr, deg_params)
                loss = loss / len(sr)
            else:
                loss = self.loss(sr, hr, deg_params)

            if loss.item() < self.args.skip_threshold * self.error_last:  # Loss is smaller than last, can update
                loss.requires_grad_(True)  # Set requires_grad to True
                loss.backward()  # Auto-compute gradients for tensors with requires_grad=True
                self.optimizer.step()  # Update parameters
            else:  # Skip this batch
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()
            # Batch index starts from 0, +1 for 1-based, *batch_size for trained data count
            if (batch + 1) * self.args.batch_size % self.args.print_every == 0:
                self.ckp.write_log('==> [{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),  # Model runtime
                    timer_data.release()))  # Data loading time

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))  # End batch training log
        self.error_last = self.loss.log[-1, -1]  # Last loss value of last batch for evaluation

        torch.cuda.synchronize()
        # end = time.time()
        # print("running time is", end - start)

    def prepare(self, l):  # Data preprocessing, modify precision
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]  # Recursive function

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

