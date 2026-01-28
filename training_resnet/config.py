from pathlib import Path


class R3DTransferConfig:
    DATASET_PATH = Path("../../Datasets")
    DATASET_NAME = "Hockey"

    VIOLENCE_PATH = DATASET_PATH / DATASET_NAME / "Violence"
    NON_VIOLENCE_PATH = DATASET_PATH / DATASET_NAME / "NonViolence"

    SPLIT_RATIO = 0.75
    N_FRAMES = 16

    BATCH_SIZE = 32
    NUM_EPOCHS = 100

    OPTIMIZER = "adamw"
    BACKBONE_LR = 1e-5
    HEAD_LR = 1e-4
    WEIGHT_DECAY = 1e-2
    BETAS = (0.9, 0.999)
    EPS = 1e-8

    FREEZE_LAYERS = ['stem', 'layer1', 'layer2']
    UNFREEZE_EPOCH = 20

    NUM_WORKERS = 0
    PIN_MEMORY = False

    EARLY_STOPPING_PATIENCE = 15

    DROPOUT_P = 0.5
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP = 1.0

    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"
    T_0 = 10
    T_MULT = 2
    ETA_MIN = 1e-7

    SAVE_DIR = Path("./heatmap_models/checkpoints_r3d18_hockey")
    MODEL_NAME = "r3d18_violence"

    HEATMAP_MODEL_BASE_DIR = Path("./heatmap_models")
    SMARTCROP_MODEL_BASE_DIR = Path("./models")

    DEVICE = "cuda"

    KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
    KINETICS_STD = [0.22803, 0.22145, 0.216989]

    USE_PRETRAINED = True

    USE_SMART_CROP = False
    PRETRAINED_MODEL_PATH = None
    SMART_CROP_PROB = 0.8
    SMART_CROP_THRESHOLD = 0.6

    AVAILABLE_DATASETS = {
        'Crowd': {
            'path': DATASET_PATH / 'Crowd',
            'violence': DATASET_PATH / 'Crowd' / 'Violence',
            'non_violence': DATASET_PATH / 'Crowd' / 'NonViolence'
        },
        'Hockey': {
            'path': DATASET_PATH / 'Hockey',
            'violence': DATASET_PATH / 'Hockey' / 'Violence',
            'non_violence': DATASET_PATH / 'Hockey' / 'NonViolence'
        },
        'Movies': {
            'path': DATASET_PATH / 'Movies',
            'violence': DATASET_PATH / 'Movies' / 'Violence',
            'non_violence': DATASET_PATH / 'Movies' / 'NonViolence'
        },
        'RLVS': {
            'path': DATASET_PATH / 'RLVS',
            'violence': DATASET_PATH / 'RLVS' / 'Violence',
            'non_violence': DATASET_PATH / 'RLVS' / 'NonViolence'
        }
    }

    def __init__(self, dataset_name='Crowd', use_smart_crop=False):
        self.use_smart_crop = use_smart_crop
        self.set_dataset(dataset_name)
        self.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    def set_dataset(self, dataset_name):
        if self.use_smart_crop:
            base_dir = self.SMARTCROP_MODEL_BASE_DIR
            suffix = "_smartcrop"
        else:
            base_dir = self.HEATMAP_MODEL_BASE_DIR
            suffix = ""

        if dataset_name == 'Mix':
            self.DATASET_NAME = 'Mix'
            self.VIOLENCE_PATH = None
            self.NON_VIOLENCE_PATH = None
            self.SAVE_DIR = base_dir / f"checkpoints_r3d18_mix{suffix}"
            self.MODEL_NAME = f"r3d18_violence_mix{suffix}"
        elif dataset_name in self.AVAILABLE_DATASETS:
            self.DATASET_NAME = dataset_name
            dataset_info = self.AVAILABLE_DATASETS[dataset_name]
            self.VIOLENCE_PATH = dataset_info['violence']
            self.NON_VIOLENCE_PATH = dataset_info['non_violence']
            self.SAVE_DIR = base_dir / f"checkpoints_r3d18_{dataset_name.lower()}{suffix}"
            self.MODEL_NAME = f"r3d18_violence_{dataset_name.lower()}{suffix}"
        else:
            raise ValueError(
                f"Dataset {dataset_name} not found. Available: {list(self.AVAILABLE_DATASETS.keys()) + ['Mix']}")

    def get_mix_paths(self):
        violence_paths = []
        non_violence_paths = []

        for dataset_name in ['Crowd', 'Hockey', 'Movies']:
            if dataset_name in self.AVAILABLE_DATASETS:
                dataset_info = self.AVAILABLE_DATASETS[dataset_name]
                violence_paths.append(dataset_info['violence'])
                non_violence_paths.append(dataset_info['non_violence'])

        return violence_paths, non_violence_paths

    def get_heatmap_model_path(self, dataset_name):
        if dataset_name == 'Mix':
            model_dir = self.HEATMAP_MODEL_BASE_DIR / "checkpoints_r3d18_mix"
            model_name = "r3d18_violence_mix_best.pth"
        else:
            model_dir = self.HEATMAP_MODEL_BASE_DIR / f"checkpoints_r3d18_{dataset_name.lower()}"
            model_name = f"r3d18_violence_{dataset_name.lower()}_best.pth"

        return model_dir / model_name

    def get_smartcrop_model_path(self, dataset_name):
        if dataset_name == 'Mix':
            model_dir = self.SMARTCROP_MODEL_BASE_DIR / "checkpoints_r3d18_mix_smartcrop"
            model_name = "r3d18_violence_mix_smartcrop_best.pth"
        else:
            model_dir = self.SMARTCROP_MODEL_BASE_DIR / f"checkpoints_r3d18_{dataset_name.lower()}_smartcrop"
            model_name = f"r3d18_violence_{dataset_name.lower()}_smartcrop_best.pth"

        return model_dir / model_name