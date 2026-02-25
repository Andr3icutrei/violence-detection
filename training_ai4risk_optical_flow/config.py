from pathlib import Path


class X3DConfig:
    DATASET_PATH = Path("../../Datasets")
    DATASET_NAME = "AI4RiSK_FLOW"

    VIOLENCE_PATH = None
    NON_VIOLENCE_PATH = None

    SPLIT_RATIO = 0.8
    X3D_VERSION = "m"

    NUM_FRAMES = 16
    TEMPORAL_STRIDE = 2

    INPUT_SIZE = 224

    BATCH_SIZE = 8
    NUM_EPOCHS = 50

    ACCUMULATION_STEPS = 3
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS

    OPTIMIZER = "adamw"
    BACKBONE_LR = 1e-5
    HEAD_LR = 1e-4
    WEIGHT_DECAY = 1e-4
    BETAS = (0.9, 0.999)
    EPS = 1e-8

    FREEZE_BACKBONE = False
    UNFREEZE_EPOCH = 5

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

    SAVE_DIR = Path("checkpoints_x3d_flow")
    MODEL_NAME = "x3d_flow_npy"

    DEVICE = "cuda"
    INPUT_CHANNELS = 2

    KINETICS_MEAN = [0.5, 0.5]
    KINETICS_STD = [0.225, 0.225]

    USE_PRETRAINED = True

    AVAILABLE_DATASETS = {
        'AI4RiSK_FLOW': {
            'path': DATASET_PATH / 'AI4RiSK_FLOW',
            'non_violence_dirs': ['0'],
            'violence_dirs': ['1', '2', '3', '4'],
            'extension': '.npy'
        },
    }

    def __init__(self):
        self.set_dataset(self.DATASET_NAME)
        self.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    def set_dataset(self, dataset_name):
        if dataset_name in self.AVAILABLE_DATASETS:
            self.DATASET_NAME = dataset_name
            dataset_info = self.AVAILABLE_DATASETS[dataset_name]
            self.VIOLENCE_PATH = dataset_info
            self.NON_VIOLENCE_PATH = dataset_info
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")