# src/config.py

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
_C.DATASET.NAME = 'CIFAR100'
_C.DATASET.DATA_DIR = '/path/to/cifar100'
_C.DATASET.IMAGE_SIZE = 224
_C.DATASET.MEAN = [0.5071, 0.4867, 0.4408]
_C.DATASET.STD = [0.2675, 0.2565, 0.2761]

_C.NUM_STAGES = 3

_C.MG_GRAPH = CN()
_C.MG_GRAPH.DEPTH = 2
_C.MG_GRAPH.LABELS = ['dog', 'cat', 'eagle', 'sparrow', 'rose', 'oak', 'mammal', 'bird', 'flower', 'tree', 'animal', 'plant']

_C.MODEL = CN()
_C.MODEL.PROMPT_DIM = 768  # Should match the hidden size of Swin Transformer
_C.MODEL.CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
_C.MODEL.PRETRAINED = True
_C.MODEL.MODEL_PATH = './logs/multi_stage_cot.pt'  # Path to saved model for evaluation

_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = 'adamw'
_C.OPTIMIZER.LR = 1e-4
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4
# _C.OPTIMIZER.MOMENTUM = 0.9  # Only needed for SGD

_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = 'step'  # 'step', 'cosine', 'cosine_restart'
_C.SCHEDULER.STEP_SIZE = 30
_C.SCHEDULER.GAMMA = 0.1
# For cosine scheduler:
# _C.SCHEDULER.T_MAX = 100
# For cosine_restart:
# _C.SCHEDULER.T_0 = 10
# _C.SCHEDULER.T_MULT = 2

_C.LOSS = CN()
_C.LOSS.ALPHA = [1.0, 1.0, 1.0]  # Weights for L_cls per stage
_C.LOSS.BETA = [0.0, 1.0, 1.0]   # Weights for L_coh per stage (start from stage 2)
_C.LOSS.GAMMA = [1.0, 1.0, 1.0]  # Weights for L_eval per stage
_C.LOSS.LAMBDA_EVAL = 1.0        # Weight for evaluator loss

_C.LOG_DIR = './logs'
_C.NUM_EPOCHS = 100
_C.BATCH_SIZE = 64
_C.NUM_WORKERS = 4
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
