import os
from importlib import import_module

import torch
import torch.nn as nn  # Neural network module


class Model(nn.Module):
    # Constructor for model initialization
    def __init__(self, args, ckp):  # ckp is a checkpoint object containing epoch and metrics
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble  # Whether to use self-ensemble technique
        self.chop = args.chop  # Whether to use patch-based processing
        self.precision = args.precision  # Whether to use mixed precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs  # Number of GPUs
        self.save_models = args.save_models

        # Load the overall model
        module = import_module('models.' + args.model.lower())
        # Call make_model to return the model, initialize it, and load to GPU
        # The instantiated object is assigned to self.model
        # 1. Original loaded model
        self.model = module.make_model(args).to(self.device)
        # 2. Model after optional half precision reduction
        if args.precision == 'half': self.model.half()  # Half precision 16-bit for faster GPU computation
        # 3. Model after optional parallelization
        if not args.cpu and args.n_GPUs > 1:
            # Parallelize the model for subsequent training and testing
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        # Load saved model weights using custom load function
        self.load(
            ckp.dir,  # Model save path
            pre_train=args.pre_train,  # If True, load weights from pretrained model
            resume=args.resume,  # If True, resume from latest checkpoint
            cpu=args.cpu  # Whether to load model to CPU
        )
        if args.print_model:
            print(self.model)

        # Forward propagation

    def forward(self, x, idx_scale=0, return_degradation=False):
        self.idx_scale = idx_scale  # Input data scale factor
        target = self.get_model()  # Get model object
        if hasattr(target, 'set_scale'):  # Check if target has set_scale attribute
            target.set_scale(idx_scale)
        # self_ensemble: perform self-ensemble by scaling and reconstructing input image multiple times
        # then averaging to reduce artifacts and noise
        # self.training indicates not in training mode (i.e., testing mode)
        # 1. For self-ensemble, forward_x8 is the ensemble version of forward
        if self.self_ensemble and not self.training:    # Testing phase
            if self.chop:  # Need to chop image edges for multiple local scalings
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward
            # forward_x8 implements self-ensemble by:
            # horizontal flip, vertical flip, diagonal flip, horizontal+vertical mirror flip
            # then forward computation and averaging results
            return self.forward_x8(x, forward_function)
        # self.ensemble and self.chop are both ensemble methods, but:
        # former uses different training datasets, latter uses different cropping on same dataset
        # 2. For chop ensemble, forward_chop is the chop version of forward
        elif self.chop and not self.training:
            return self.forward_chop(x)
        # 3. In training phase, neither if is executed, simply return model output
        else:
            if return_degradation:
                return self.model(x, return_degradation=True)
            return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            # When GPUs > 1, self.model is nn.DataParallel object
            # Need to access .module for actual model object
            return self.model.module

    def state_dict(self, **kwargs):  # Save model state
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):  # apath is save path root directory
        target = self.get_model()
        # torch.save(
        #    target.state_dict(),
        #    os.path.join(apath, 'model', 'model_latest.pt')
        # )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        # pre_train is optional pretrained model path, . means current path (no pretrained model)
        # resume specifies model state to load: -1=latest, 0=pretrained, >0=historical state
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':  # pre_train != . means pretrained model exists
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                    # strict=False allows loading with architecture mismatches for better predictions
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )

    def forward_chop(self, x, shave=10, min_size=160000):  # shave is reserved edge pixel size
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()  # batch_size, channel, height, width
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],  # Top-left part
            x[:, :, 0:h_size, (w - w_size):w],  # Top-right part
            x[:, :, (h - h_size):h, 0:w_size],  # Bottom-left part
            x[:, :, (h - h_size):h, (w - w_size):w]]  # Bottom-right part

        if w_size * h_size < min_size:  # If cropped image is smaller than min, process directly
            sr_list = []  # Store processed images
            for i in range(0, 4, n_GPUs):
                # Concatenate image patches along batch dimension (dim=0), batch size = n_GPUs
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                # Pass each batch through model
                sr_batch = self.model(lr_batch)
                # Split processed batch back into individual patches
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:  # If cropped size >= preset, recursively call forward_chop
            # Backslash concatenates all processed patches
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)  # Create new image same size as original for pixel filling
        # Fill each small pixel block from first large patch to top-left of output
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()
            # Convert non-single precision tensor to single precision
            # Copy converted data from GPU to memory and convert to numpy array
            v2np = v.data.cpu().numpy()
            if op == 'v':  # Vertical flip
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':  # Horizontal flip
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':  # Transpose
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()  # Convert to half precision float

            return ret

        # Apply transform functions defined above for series of flip operations
        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        # Call forward for SR processing
        sr_list = [forward_function(aug) for aug in lr_list]
        # Apply vertical/horizontal rotations to processed SR images based on index
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)  # Average all processed SR images

        return output
