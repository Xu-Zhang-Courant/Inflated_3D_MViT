class CfgNode:
    def __init__(self):
        pass

class cfg:
    AUG = CfgNode()
    AUG.NUM_SAMPLE = 1
    AUG.COLOR_JITTER = 0.4
    AUG.AA_TYPE = "rand-m9-n6-mstd0.5-inc1"
    AUG.INTERPOLATION = "bicubic"
    AUG.RE_PROB = 0.25
    AUG.RE_MODE = "pixel"
    AUG.RE_COUNT = 1
    AUG.RE_SPLIT = False

    MIXUP = CfgNode()
    MIXUP.ENABLE = True
    MIXUP.ALPHA = 0.8
    MIXUP.CUTMIX_ALPHA = 1.0
    MIXUP.PROB = 1.0
    MIXUP.SWITCH_PROB = 0.5
    MIXUP.LABEL_SMOOTH_VALUE = 0.1

    TEST = CfgNode()
    TEST.ENABLE = False
    TEST.DATASET = "imagenet"
    TEST.BATCH_SIZE = 64
    TEST.CHECKPOINT_FILE_PATH = ""
    TEST.CHECKPOINT_SQUEEZE_TEMPORAL = True

    MODEL = CfgNode()
    MODEL.MODEL_NAME = "MViT"
    MODEL.NUM_CLASSES = 1000
    MODEL.LOSS_FUNC = "soft_cross_entropy"
    MODEL.DROPOUT_RATE = 0.0
    MODEL.HEAD_ACT = "softmax"
    MODEL.ACT_CHECKPOINT = False

    MVIT = CfgNode()
    MVIT.MODE = "conv"
    MVIT.POOL_FIRST = False
    MVIT.CLS_EMBED_ON = False
    MVIT.PATCH_KERNEL = [7, 7]
    MVIT.PATCH_STRIDE = [4, 4]
    MVIT.PATCH_PADDING = [3, 3]
    MVIT.EMBED_DIM = 96
    MVIT.NUM_HEADS = 1
    MVIT.MLP_RATIO = 4.0
    MVIT.QKV_BIAS = True
    MVIT.DROPPATH_RATE = 0.1
    MVIT.DEPTH = 10

    MVIT.POOL_KV_STRIDE = None


    MVIT.DIM_MUL =[[1, 2.0], [3, 2.0], [8, 2.0]]
    MVIT.HEAD_MUL= [[1, 2.0], [3, 2.0], [8, 2.0]]
    MVIT.POOL_KVQ_KERNEL= [3, 3]
    MVIT.POOL_KV_STRIDE_ADAPTIVE=[4, 4]
    MVIT.POOL_Q_STRIDE= [[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 2, 2],[9, 1, 1]]
    MVIT.ZERO_DECAY_POS_CLS = False
    MVIT.USE_ABS_POS = False
    MVIT.REL_POS_SPATIAL = True
    MVIT.REL_POS_ZERO_INIT = False
    MVIT.RESIDUAL_POOLING = True
    MVIT.DIM_MUL_IN_ATT = True

    DATA = CfgNode()
    # The spatial crop size for training.
    DATA.TRAIN_CROP_SIZE = 224
    # The spatial crop size for testing.
    DATA.TEST_CROP_SIZE = 224