from importlib import import_module

from torch.utils.data.dataloader import default_collate  # Auto-collate data by sample type

from mydata.myDataLoader import MSDataLoader

class Data:
    def __init__(self, args):
        """Construct and return data loaders.
        :param args: args
        """

        kwargs = {}
        if not args.cpu:  # Use GPU for computation
            kwargs['collate_fn'] = default_collate  # Collate data into a batch
            kwargs['pin_memory'] = True  # Pin memory for faster GPU data transfer
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        '''Load trainset and construct trainLoader'''
        self.loader_train = None  # Store loaded dataset
        # if not test only
        if not args.test_only:
            # 1. Dynamically import the corresponding dataset module
            module_train = import_module('mydata.' + args.data_train.lower())  # Lowercase because mydata module names are lowercase
            # 2. Generate trainset
            # getattr retrieves the dataset class from module_train, (args) initializes it
            trainset = getattr(module_train, args.data_train)(args)
            # 3. Load trainset
            self.loader_train = MSDataLoader(  # Training data loader
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,  # Shuffle data
                **kwargs  # Pass other kwargs including num_workers, pin_memory to parent constructor
            )

        '''Load testset and construct testLoader'''
        if args.data_test in ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']:
            if not args.benchmark_noise:  # Not a standard noisy benchmark dataset
                module_test = import_module('mydata.benchmark')  # Get benchmark module
                # Load the Benchmark class from the module
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('mydata.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )
        else:  # If test dataset is not in the five standard sets, load based on parameter
            module_test = import_module('mydata.' + args.data_test.lower())  # Import based on runtime parameter
            testset = getattr(module_test, args.data_test)(args, train=False)
        self.loader_test = MSDataLoader(  # Test data loader
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )
