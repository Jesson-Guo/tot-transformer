import os
import yaml
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
_C.DATASET.NUM_CLASSES = 100
_C.DATASET.NAME = 'CIFAR100'
_C.DATASET.DATA_PATH = '/path/to/cifar100'
_C.DATASET.HIERARCHY = '/path/to/hierarchy'
_C.DATASET.IMAGE_SIZE = 224
_C.DATASET.INTERPOLATION = 'bicubic'
_C.DATASET.PIN_MEMORY = True

# Model
_C.MODEL = CN()
_C.MODEL.MODEL_PATH = ''  # Path to saved model for evaluation
_C.MODEL.BACKBONE_ROOT = ''
_C.MODEL.CLIP_ROOT = ''
_C.MODEL.EMBED_DIMS = [256, 512]
_C.MODEL.NUM_HEADS = [8, 16]
_C.MODEL.MLP_RATIOS = [4, 2]
_C.MODEL.NUM_DECODER_LAYERS = 4

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "swin"
_C.MODEL.BACKBONE.EMBED_DIMS = [64, 128, 256, 512]
_C.MODEL.BACKBONE.NUM_HEADS = [4, 8, 16, 32]
_C.MODEL.BACKBONE.DEPTHS = [2, 2, 18, 2]
_C.MODEL.BACKBONE.WINDOW_SIZE = 7
_C.MODEL.BACKBONE.PATCH_SIZE = 7
_C.MODEL.BACKBONE.MLP_RATIOS = [8, 6, 4, 2]

# SMT parameters
_C.MODEL.BACKBONE.IN_CHANS = 7
_C.MODEL.BACKBONE.CA_NUM_HEADS = [4, 4, 4, -1]
_C.MODEL.BACKBONE.SA_NUM_HEADS = [-1, -1, 8, 16]
_C.MODEL.BACKBONE.QKV_BIAS = True
_C.MODEL.BACKBONE.QK_SCALE = None
_C.MODEL.BACKBONE.PROJ_DROP_RATE = 0.0
_C.MODEL.BACKBONE.ATTN_DROP_RATE = 0.0
_C.MODEL.BACKBONE.DROP_PATH_RATE = 0.1
_C.MODEL.BACKBONE.LAYERSCALE_VALUE = 1e-4
_C.MODEL.BACKBONE.USE_LAYERSCALE = False
_C.MODEL.BACKBONE.CA_ATTENTIONS = [1, 1, 1, 0]
_C.MODEL.BACKBONE.EXPAND_RATIO = 2

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
_C.LOSS.LAMBDA_MERO = 1.0
_C.LOSS.EOS_COEF = 0.1
_C.LOSS.LAMBDA_BASE = 1.0
_C.LOSS.LAMBDA_COH = 1.0

_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.RESUME = ''

_C.LOG_DIR = './logs'
_C.START_EPOCH = 0
_C.NUM_EPOCHS = 100
_C.BATCH_SIZE = 64
_C.NUM_WORKERS = 8
_C.LABEL_SMOOTHING = 0.1
_C.AMP_ENABLE = True


def update_config_from_file(config, cfg_file):
    """
    Update config based on config file.
    Args:
        config: CfgNode object
        cfg_file: config file path
    """
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def load_config(cfg_file):
    """
    Loads the YAML configuration file.
    """
    config = _C.clone()
    update_config_from_file(config, cfg_file)
    return config
