import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class MSDataLoader(DataLoader):
    """Multi-scale DataLoader compatible with modern PyTorch versions.
    
    This loader handles multi-scale training by randomly selecting a scale
    for each batch and appending the scale index to the batch data.
    
    Note: For multi-scale training (len(scale) > 1), num_workers is set to 0
    to ensure scale changes are visible to the dataset during data loading.
    For single-scale training, multi-threading works normally.
    """
    
    def __init__(self, args, dataset, batch_size=1, shuffle=False,
                 sampler=None, batch_sampler=None,
                 collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        
        self.scale = args.scale
        self._ms_dataset = dataset  # Store reference for multi-scale handling
        
        # For multi-scale training, we need single-threaded loading
        # so that set_scale() is visible to __getitem__
        num_workers = args.n_threads
        if len(self.scale) > 1 and hasattr(dataset, 'train') and dataset.train:
            num_workers = 0
        
        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

    def __iter__(self):
        # For single-scale, just iterate normally and append scale index
        if len(self.scale) == 1:
            for batch in super().__iter__():
                if isinstance(batch, (list, tuple)):
                    yield list(batch) + [0]
                else:
                    yield batch
        else:
            # For multi-scale, set scale before each batch
            # This works because num_workers=0 for multi-scale
            batch_sampler = self.batch_sampler
            for batch_indices in batch_sampler:
                idx_scale = 0
                if hasattr(self._ms_dataset, 'train') and self._ms_dataset.train:
                    idx_scale = random.randrange(0, len(self.scale))
                    self._ms_dataset.set_scale(idx_scale)
                
                # Manually load and collate the batch
                batch = [self._ms_dataset[i] for i in batch_indices]
                batch = self.collate_fn(batch)
                
                if isinstance(batch, (list, tuple)):
                    yield list(batch) + [idx_scale]
                else:
                    yield batch
