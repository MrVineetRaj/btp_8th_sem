import gc
import torch
import loss
import models
import mydata
import utility
from option import args
from trainer import Trainer

# from torchstat import stat
# import torchvision.models as models

torch.manual_seed(args.seed)  # Set random seed for reproducibility
checkpoint = utility.checkpoint(args)  # Checkpoint stores training progress, model params, etc.
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,2,3,5'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Automatic memory management, force garbage collection
gc.collect()
# Clear GPU memory cache
torch.cuda.empty_cache()
# Check if previously trained model is available
if checkpoint.ok:
    loader = mydata.Data(args)  # Create data loader
    # Create data loaders (both train and test loaders) for the model
    # print(loader)
    model = models.Model(args, checkpoint)  # Initialize model with checkpoint params or default
    # Skip loss computation if only testing
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    # Pass data loader, model, loss function to trainer
    t = Trainer(args, loader, model, loss, checkpoint)  # Instantiate trainer
    while not t.terminate():
        # args.test_only = False
        t.train()  # Model training
        # args.test_only = True
        # print("test_only_before:", args.test_only)  # False
        total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters:", total_params)
        t.test()  # Model testing
        # print("test_only_after:",args.test_only)
    #  stat(model,(3,64,64))
    checkpoint.done()  # Save training results to checkpoint after training ends

