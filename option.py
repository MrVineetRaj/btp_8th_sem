import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',  # Debug switch
                    help='Enables debug mode')
parser.add_argument('--template', default='.',  # Set various templates in option.py
                    help='You can set various templates in option.py')

# Hardware settings
parser.add_argument('--n_threads', type=int, default=2,  # Number of threads for data loading (2 for Colab)
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',  # Use CPU only
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,  # Number of GPUs
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,  # Random seed
                    help='random seed')

# Data settings
parser.add_argument('--dir_data', type=str, default='./data/',  # Dataset directory
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../DPDNN',  # Demo image directory
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',  # Training dataset name
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',  # Test dataset name
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',  # Use noisy benchmark sets
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=300,  # Training set size (300 images from DIV2K)
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=200,  # Validation set size (200 images)
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=300,  # Validation starts at image 301
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='img',  # Dataset file extension
                    help='dataset file extension')
parser.add_argument('--scale', default='2',  # Super-resolution scale
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=96,  # Output patch size (96 for efficient training)
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,  # Maximum RGB value
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,  # Number of color channels
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',  # Gaussian noise standard deviation
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',  # Enable memory-efficient forward
                    help='enable memory-efficient forward')

# Degradation settings for real-world robustness
parser.add_argument('--use_degradation', action='store_true',
                    help='Enable realistic degradation augmentation during training')
parser.add_argument('--blur_sigma_range', type=str, default='0.2,3.0',
                    help='Gaussian blur sigma range (min,max)')
parser.add_argument('--jpeg_quality_range', type=str, default='30,95',
                    help='JPEG compression quality range (min,max)')
parser.add_argument('--degradation_prob', type=float, default=0.5,
                    help='Probability of applying each degradation type')
parser.add_argument('--kernel_size', type=int, default=21,
                    help='Blur kernel size for degradation estimation')
parser.add_argument('--deg_loss_weight', type=float, default=0.1,
                    help='Weight for degradation estimation loss')

# Model settings
parser.add_argument('--model', default='EPGDUN',  # Model name
                    help='model name')

parser.add_argument('--act', type=str, default='relu',  # Activation function
                    help='activation function')
parser.add_argument('--precision', type=str, default='Single',  # Precision
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Trainer settings
parser.add_argument('--reset', action='store_true',  # Reset training
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,  # Test every N batches
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=200,  # Number of epochs to train
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=4,  # Training batch size
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,  # Split batch into smaller chunks
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',  # Use self-ensemble for testing
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',  # Set to test mode only
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,  # k value for adversarial loss
                    help='k value for adversarial loss')

# Optimization settings
parser.add_argument('--lr', type=float, default=1e-4,  # Learning rate
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=400,  # Learning rate decay every N epochs
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step_100_150',  # Learning rate decay at epochs 100 and 150
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,  # Learning rate decay factor for step decay
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),  # Optimizer to use
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,  # SGD momentum
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,  # ADAM beta1
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,  # ADAM beta2
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,  # ADAM epsilon for numerical stability
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,  # Weight decay
                    help='weight decay')

# Loss settings
parser.add_argument('--loss', type=str, default='1*L1',  # Loss function configuration
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e10',  # Skip batches with large error
                    help='skipping batch that has large error')

# Log settings
parser.add_argument('--pre_train', type=str, default='.',  # Pretrained model path (. means no pretrain)
                    help='pre-trained model directory')
parser.add_argument('--save', type=str, default='../experiment/SR_X2_BI',  # Filename for saving
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',  # Filename for loading
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,  # Resume from specific checkpoint
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',  # Print model
                    help='print model')
parser.add_argument('--save_models', action='store_true', default=True,  # Save all intermediate models
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=50,  # Batches before logging training status
                    help='how many batches to wait before logging training status')
# parser.add_argument('--save_results', action='store_true',
parser.add_argument('--save_results', type=bool, default=True,  # Save results
                    help='save output results')

# Options for residual group and feature channel reduction
parser.add_argument('--n_resgroups', type=int, default=10,  # Number of residual groups
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,  # Feature map reduction
                    help='number of feature maps reduction')
# Options for test
parser.add_argument('--testset', type=str, default='DHT',  # Dataset name for testing
                    help='dataset name for testing')
# Above code adds command-line options
args = parser.parse_args()  # Parse command-line args into Python objects
template.set_template(args)

args.scale = list(
    map(lambda x: int(x), args.scale.split('+'))
)
# args.scale.split('+') splits the scale string e.g. "2+3+4" by '+' into a list of substrings,
# then converts each string element to integer using lambda, and creates an integer list
if args.epochs == 0:
    args.epochs = 1e8
# Convert string boolean values to actual boolean type
# vars(args) returns the argument dictionary, vars(args)[arg] returns the value
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

