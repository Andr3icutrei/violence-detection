from pathlib import Path


class DenseNet3DConfig:
    DATASET_PATH = Path("../../Datasets")
    DATASET_NAME = "Mix"

    VIOLENCE_PATH = DATASET_PATH / DATASET_NAME / "Violence"
    NON_VIOLENCE_PATH = DATASET_PATH / DATASET_NAME / "NonViolence"

    SPLIT_RATIO = 0.75
    N_FRAMES = 16

    BATCH_SIZE = 8
    NUM_EPOCHS = 100

    OPTIMIZER = "adamw"
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.5
    WEIGHT_DECAY = 1e-3
    BETAS = (0.9, 0.999)
    EPS = 1e-8

    NUM_WORKERS = 0
    PIN_MEMORY = False

    EARLY_STOPPING_PATIENCE = 20

    DROPOUT_P = 0.5
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP = 1.0

    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"
    T_0 = 10
    T_MULT = 2
    ETA_MIN = 1e-7

    SAVE_DIR = Path("checkpoints_densenet3d_mix")
    MODEL_NAME = "densenet3d_violence"

    DEVICE = "cuda"

    KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
    KINETICS_STD = [0.22803, 0.22145, 0.216989]

    GROWTH_RATE = 32
    BLOCK_CONFIG = (6, 12, 24)
    NUM_INIT_FEATURES = 64

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
    }

    def __init__(self, dataset_name='Crowd'):
        self.set_dataset(dataset_name)
        self.SAVE_DIR.mkdir(exist_ok=True)

    def set_dataset(self, dataset_name):
        if dataset_name == 'Mix':
            self.DATASET_NAME = 'Mix'
            self.VIOLENCE_PATH = None
            self.NON_VIOLENCE_PATH = None
            self.SAVE_DIR = Path(f"checkpoints_densenet3d_mix")
            self.MODEL_NAME = "densenet3d_violence_mix"
        elif dataset_name in self.AVAILABLE_DATASETS:
            self.DATASET_NAME = dataset_name
            dataset_info = self.AVAILABLE_DATASETS[dataset_name]
            self.VIOLENCE_PATH = dataset_info['violence']
            self.NON_VIOLENCE_PATH = dataset_info['non_violence']
            self.SAVE_DIR = Path(f"checkpoints_densenet3d_{dataset_name.lower()}")
            self.MODEL_NAME = f"densenet3d_violence_{dataset_name.lower()}"
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