from pathlib import Path


class SlowFastConfig:
    DATASET_PATH = Path("../../Datasets")
    DATASET_NAME = "AI4RiSK_CROPPED_SR_V2"

    VIOLENCE_PATH = None
    NON_VIOLENCE_PATH = None

    SEED = 42

    SPLIT_RATIO = 0.8

    NUM_CLASSES = 5
    CLASS_NAMES = ['Non-Violence', 'Shooting', 'Throwing', 'Punching', 'Running/Pushing']

    SLOWFAST_ALPHA = 4
    SLOWFAST_BETA = 0.125

    SLOW_FRAMES = 8
    FAST_FRAMES = 32
    TEMPORAL_STRIDE = 1

    INPUT_SIZE = 224
    CROP_SIZE = 224

    USE_CROP = False

    BATCH_SIZE = 8
    NUM_EPOCHS = 100

    ACCUMULATION_STEPS = 4
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS

    OPTIMIZER = "adamw"
    BACKBONE_LR = 1e-5
    HEAD_LR = 1e-4
    WEIGHT_DECAY = 1e-2
    BETAS = (0.9, 0.999)
    EPS = 1e-8

    FREEZE_BACKBONE = True
    UNFREEZE_EPOCH = 20

    NUM_WORKERS = 4
    PIN_MEMORY = True

    EARLY_STOPPING_PATIENCE = 15

    DROPOUT_P = 0.5
    LABEL_SMOOTHING = 0.0
    GRAD_CLIP = 5.0

    USE_FOCAL_LOSS = False
    FOCAL_GAMMA = 2.0
    USE_CLASS_WEIGHTS = True
    USE_BALANCED_SAMPLING = False

    USE_AMP = True

    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"
    T_0 = 10
    T_MULT = 2
    ETA_MIN = 1e-7

    SAVE_DIR = Path("checkpoints_slowfast_ai4risk")
    MODEL_NAME = "slowfast_violence_ai4risk"

    DEVICE = "cuda"

    KINETICS_MEAN = [0.45, 0.45, 0.45]
    KINETICS_STD = [0.225, 0.225, 0.225]

    USE_PRETRAINED = True

    AVAILABLE_DATASETS = {
        'AI4RiSK': {
            'path': DATASET_PATH / 'AI4RiSK_CROPPED_SR_V2',
            'non_violence_dirs': ['0'],
            'violence_dirs': ['1', '2', '3', '4'],
            'type': 'multiclass'
        },
    }

    def __init__(self):
        self.set_dataset('AI4RiSK')
        self.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    def set_dataset(self, dataset_name):
        if dataset_name == 'AI4RiSK':
            self.DATASET_NAME = 'AI4RiSK'
            dataset_info = self.AVAILABLE_DATASETS['AI4RiSK']
            self.VIOLENCE_PATH = dataset_info
            self.NON_VIOLENCE_PATH = dataset_info
            self.SAVE_DIR = Path("checkpoints_slowfast_ai4risk")
            self.MODEL_NAME = "slowfast_violence_ai4risk"
            self.USE_CROP = False
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Only AI4RiSK is available.")