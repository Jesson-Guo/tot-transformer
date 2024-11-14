from yacs.config import CfgNode as CN

_C = CN()

# Augmentation settings
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.2  #0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0  #1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# Additional configuration parameters
_C.MODE = 'train'  # or 'eval'
_C.DATASET = CN()
_C.DATASET.NUM_CLASSES = 1000
_C.DATASET.NAME = 'CIFAR100'
_C.DATASET.DATA_PATH = '/path/to/cifar100'
_C.DATASET.IMAGE_SIZE = 224
_C.DATASET.INTERPOLATION = 'bicubic'
_C.DATASET.MEAN = [0.5071, 0.4867, 0.4408]
_C.DATASET.STD = [0.2675, 0.2565, 0.2761]
_C.DATASET.PIN_MEMORY = True

_C.NUM_STAGES = 4

_C.MG_GRAPH = CN()
_C.MG_GRAPH.DEPTH = 4
_C.MG_GRAPH.LABELS = ['dog', 'cat', 'eagle', 'sparrow', 'rose', 'oak', 'mammal', 'bird', 'flower', 'tree', 'animal', 'plant']

_C.MODEL = CN()
_C.MODEL.BACKBONE = CN()
_C.MODEL.THOUGHT_GENERATOR = CN()
_C.MODEL.STATE_EVALUATOR = CN()
_C.MODEL.PROMPT_DIM = 768  # Should match the hidden size of Swin Transformer
_C.MODEL.CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
_C.MODEL.PRETRAINED = True
_C.MODEL.MODEL_PATH = './logs/multi_stage_cot.pt'  # Path to saved model for evaluation

# Optimizer
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.BASE_LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 0.05

# LR scheduler
_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.SCHEDULER.GAMMA = 0.1
_C.SCHEDULER.MULTISTEPS = []
_C.SCHEDULER.WARMUP_EPOCHS = 5
_C.SCHEDULER.WEIGHT_DECAY = 0.05
_C.SCHEDULER.WARMUP_LR = 1e-7
_C.SCHEDULER.MIN_LR = 1e-6

_C.LOSS = CN()
_C.LOSS.ALPHA = [1.0, 1.0, 1.0]  # Weights for L_cls per stage
_C.LOSS.BETA = [0.0, 1.0, 1.0]   # Weights for L_coh per stage (start from stage 2)
_C.LOSS.GAMMA = [1.0, 1.0, 1.0]  # Weights for L_eval per stage
_C.LOSS.LAMBDA_EVAL = 1.0        # Weight for evaluator loss

_C.LOG_DIR = './logs'
_C.START_EPOCH = 0
_C.NUM_EPOCHS = 100
_C.BATCH_SIZE = 64
_C.NUM_WORKERS = 8
_C.LOG_INTERVAL = 100


def update_config(config, args):
    """
    Update config based on command line arguments.
    Args:
        config: CfgNode object
        args: argparse.Namespace
    """
    # Override config options with command line arguments
    # Assuming args are provided in the format key=value, e.g., MODEL.PRETRAINED=False

    for key, value in vars(args).items():
        if key in config:
            config[key] = value
        else:
            # Handle nested keys separated by dots
            keys = key.split('.')
            cfg = config
            for sub_key in keys[:-1]:
                if sub_key not in cfg:
                    cfg[sub_key] = CN()
                cfg = cfg[sub_key]
            cfg[keys[-1]] = value
