from pathlib import Path


class X3DConfig:
    DATASET_PATH = Path("../../Datasets")
    DATASET_NAME = "AI4RiSK_CROPPED_SR_V2"

    VIOLENCE_PATH = None
    NON_VIOLENCE_PATH = None

    SPLIT_RATIO = 0.8

    X3D_VERSION = "m"

    NUM_FRAMES = 16
    TEMPORAL_STRIDE = 2

    INPUT_SIZE = 224
    CROP_SIZE = 224

    USE_CROP = False

    BATCH_SIZE = 12
    NUM_EPOCHS = 100

    ACCUMULATION_STEPS = 3
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS

    OPTIMIZER = "adamw"
    BACKBONE_LR = 1e-5
    HEAD_LR = 1e-4
    WEIGHT_DECAY = 1e-3
    BETAS = (0.9, 0.999)
    EPS = 1e-8

    FREEZE_BACKBONE = False
    UNFREEZE_EPOCH = 20

    NUM_WORKERS = 4
    PIN_MEMORY = True

    EARLY_STOPPING_PATIENCE = 15

    DROPOUT_P = 0.5
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP = 2.0

    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"
    T_0 = 10
    T_MULT = 2
    ETA_MIN = 1e-7

    SAVE_DIR = Path("checkpoints_x3d_ai4risk")
    MODEL_NAME = "x3d_violence_ai4risk"

    DEVICE = "cuda"

    KINETICS_MEAN = [0.45, 0.45, 0.45]
    KINETICS_STD = [0.225, 0.225, 0.225]

    USE_PRETRAINED = True

    USE_OPTICAL_FLOW = True

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
            self.SAVE_DIR = Path("checkpoints_x3d_ai4risk")
            self.MODEL_NAME = "x3d_violence_ai4risk"
            self.USE_CROP = False
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Only AI4RiSK is available.")