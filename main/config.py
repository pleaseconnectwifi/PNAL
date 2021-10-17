"""Configuration"""
from yacs.config import CfgNode as CN
from yacs.config import load_cfg

_C = CN()

# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = "@"
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True

_C.RNG_SEED = 1

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
_C.DATA = CN()

_C.DATA.NUM_WORKERS = 1
_C.DATA.NOISE_RATE = 0.6
_C.DATA.THRESHOLD = -1.0
_C.DATA.SHUFFLE = 0
_C.DATA.EXP_NAME = ""
_C.DATA.WHICH_GPU = 0
_C.DATA.WARM_UP = 4
_C.DATA.QUEUE_SIZE = 4
_C.DATA.HAS_INST = True
_C.DATA.VOTE = 0
_C.DATA.EMA = 0
_C.DATA.EMA_DECAY = 0.999
_C.DATA.THRESHOLD_VOTING = 4
_C.DATA.LOG_LOSS_EACH_SAMPLE = 0
_C.DATA.VOTE_BEGIN = -1
_C.DATA.NET = "DGCNN"
_C.DATA.EMA_LOSSARRAY = False
_C.DATA.DATASET = "S3DIS"
_C.DATA.P_NOTUPDATE = 0.0
_C.DATA.DATA_DIR_NAME = 'individual_data'

_C.DATA.PC = CN()
_C.DATA.PC.TRAIN = CN()
_C.DATA.PC.TRAIN.INPUT_DIR = ""
_C.DATA.PC.TRAIN.GT_DIR = ""
_C.DATA.PC.TRAIN.SHAPE_LIST = ""

_C.DATA.PC.VAL = CN()
_C.DATA.PC.VAL.INPUT_DIR = ""
_C.DATA.PC.VAL.GT_DIR = ""
_C.DATA.PC.VAL.SHAPE_LIST = ""

_C.DATA.PC.TEST = CN()
_C.DATA.PC.TEST.INPUT_DIR = ""
_C.DATA.PC.TEST.GT_DIR = ""
_C.DATA.PC.TEST.SHAPE_LIST = ""


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.WEIGHT = ""
_C.MODEL.EDGE_CHANNELS = ()
_C.MODEL.LOSS_FUNCTION = ""

# ---------------------------------------------------------------------------- #
# Solver (optimizer)
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# Type of optimizer
_C.SOLVER.TYPE = "Adam"

# Basic parameters of solvers
# Notice to change learning rate according to batch size
_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.WEIGHT_DECAY = 0.0

# Specific parameters of solvers
_C.SOLVER.RMSprop = CN()
_C.SOLVER.RMSprop.alpha = 0.9

_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.momentum = 0.9

_C.SOLVER.Adam = CN()
_C.SOLVER.Adam.weight_decay = 0.0

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = "StepLR"

_C.SCHEDULER.MAX_EPOCH = 2

_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.step_size = 5
_C.SCHEDULER.StepLR.gamma = 0.9

_C.SCHEDULER.MultiStepLR = CN()
_C.SCHEDULER.MultiStepLR.milestones = ()
_C.SCHEDULER.MultiStepLR.gamma = 0.1

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 1

# The period to save a checkpoint
_C.TRAIN.CHECKPOINT_PERIOD = 1000
_C.TRAIN.LOG_PERIOD = 10
# The period to validate
_C.TRAIN.VAL_PERIOD = 0
# Data augmentation. The format is "method" or ("method", *args)
# For example, ("PointCloudRotate", ("PointCloudRotatePerturbation",0.1, 0.2))
_C.TRAIN.AUGMENTATION = ()

# Regex patterns of modules and/or parameters to freeze
# For example, ("bn",) will freeze all batch normalization layers' weight and bias;
# And ("module:bn",) will freeze all batch normalization layers' running mean and var.
_C.TRAIN.FROZEN_PATTERNS = ()

_C.TRAIN.VAL_METRIC = "l1"

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 1

# The path of weights to be tested. "@" has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TEST.WEIGHT = ""

# Data augmentation.
_C.TEST.AUGMENTATION = ()

_C.TEST.LOG_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Specific validation options
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

_C.VAL.BATCH_SIZE = 1


def load_cfg_from_file(cfg_filename):
    """Load config from a file

    Args:
        cfg_filename (str):

    Returns:
        CfgNode: loaded configuration

    """
    with open(cfg_filename, "r") as f:
        cfg = load_cfg(f)

    cfg_template = _C
    cfg_template.merge_from_other_cfg(cfg)
    return cfg_template


def get_cur_cfg():
    return _C

