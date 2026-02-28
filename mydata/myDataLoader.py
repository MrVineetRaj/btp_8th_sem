import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class MSDataLoader:
    """Multi-scale DataLoader compatible with modern PyTorch versions.
    
    Uses composition instead of inheritance to avoid PyTorch 2.x 
    DataLoader's __setattr__ restrictions.
    """
    
    def __init__(self, args, dataset, batch_size=1, shuffle=False,
                 sampler=None, batch_sampler=None,
                 collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        
        self.scale = args.scale
        self._dataset = dataset
        
        # For multi-scale training, we need single-threaded loading
        num_workers = args.n_threads
        if len(self.scale) > 1 and hasattr(dataset, 'train') and dataset.train:
            num_workers = 0
        
        # Create internal DataLoader
        self._loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def batch_sampler(self):
        return self._loader.batch_sampler
    
    @property
    def collate_fn(self):
        return self._loader.collate_fn

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        # For single-scale, just iterate normally and append scale index
        if len(self.scale) == 1:
            for batch in self._loader:
                if isinstance(batch, (list, tuple)):
                    yield list(batch) + [0]
                else:
                    yield batch
        else:
            # For multi-scale, set scale before each batch
            for batch_indices in self._loader.batch_sampler:
                idx_scale = 0
                if hasattr(self._dataset, 'train') and self._dataset.train:
                    idx_scale = random.randrange(0, len(self.scale))
                    self._dataset.set_scale(idx_scale)
                
                # Manually load and collate the batch
                batch = [self._dataset[i] for i in batch_indices]
                batch = self._loader.collate_fn(batch)
                
                if isinstance(batch, (list, tuple)):
                    yield list(batch) + [idx_scale]
                else:
                    yield batch
