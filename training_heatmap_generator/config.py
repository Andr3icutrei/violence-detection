from pathlib import Path


class R3DTransferConfig:
    DATASET_PATH = Path("../../Datasets")
    DATASET_NAME = "Crowd"

    VIOLENCE_PATH = DATASET_PATH / DATASET_NAME / "Violence"
    NON_VIOLENCE_PATH = DATASET_PATH / DATASET_NAME / "NonViolence"

    SPLIT_RATIO = 0.75
    N_FRAMES = 16

    BATCH_SIZE = 16
    NUM_EPOCHS = 100

    BACKBONE_LR = 1e-4
    HEAD_LR = 1e-3
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    FREEZE_LAYERS = ['stem', 'layer1', 'layer2']
    UNFREEZE_EPOCH = 20

    NUM_WORKERS = 0
    PIN_MEMORY = False

    EARLY_STOPPING_PATIENCE = 20

    SAVE_DIR = Path("checkpoints_r3d18")
    MODEL_NAME = "r3d18_violence"

    DEVICE = "cuda"

    KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
    KINETICS_STD = [0.22803, 0.22145, 0.216989]

    USE_PRETRAINED = True

    def __init__(self):
        self.SAVE_DIR.mkdir(exist_ok=True)