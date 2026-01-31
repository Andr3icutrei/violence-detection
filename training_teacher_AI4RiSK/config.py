from pathlib import Path


class SlowFastConfig:
    DATASET_PATH = Path("../../Datasets")
    DATASET_NAME = "Mix"

    VIOLENCE_PATH = DATASET_PATH / DATASET_NAME / "Violence"
    NON_VIOLENCE_PATH = DATASET_PATH / DATASET_NAME / "NonViolence"

    SPLIT_RATIO = 0.8

    SLOWFAST_ALPHA = 4
    SLOWFAST_BETA = 0.125

    SLOWFAST_TAU_FAST = 2
    SLOWFAST_TAU_SLOW = 16

    NUM_FRAMES_FAST = 32
    NUM_FRAMES_SLOW = NUM_FRAMES_FAST // SLOWFAST_ALPHA

    INPUT_SIZE = 224
    CROP_SIZE = 224

    USE_CROP = True

    BATCH_SIZE = 4
    NUM_EPOCHS = 100

    ACCUMULATION_STEPS = 8
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS

    OPTIMIZER = "adamw"
    BACKBONE_LR = 1e-5
    HEAD_LR = 1e-4
    WEIGHT_DECAY = 1e-2
    BETAS = (0.9, 0.999)
    EPS = 1e-8

    FREEZE_BACKBONE = False
    UNFREEZE_EPOCH = 20

    NUM_WORKERS = 4
    PIN_MEMORY = True

    EARLY_STOPPING_PATIENCE = 15

    DROPOUT_P = 0.5
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP = 1.0

    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"
    T_0 = 10
    T_MULT = 2
    ETA_MIN = 1e-7

    SAVE_DIR = Path("checkpoints_slowfast_mix")
    MODEL_NAME = "slowfast_violence"

    DEVICE = "cuda"

    KINETICS_MEAN = [0.45, 0.45, 0.45]
    KINETICS_STD = [0.225, 0.225, 0.225]

    USE_PRETRAINED = True

    INCLUDE_AI4RISK_IN_MIX = False

    AVAILABLE_DATASETS = {
        'Crowd': {
            'path': DATASET_PATH / 'Crowd',
            'violence': DATASET_PATH / 'Crowd' / 'Violence',
            'non_violence': DATASET_PATH / 'Crowd' / 'NonViolence',
            'type': 'standard'
        },
        'Hockey': {
            'path': DATASET_PATH / 'Hockey',
            'violence': DATASET_PATH / 'Hockey' / 'Violence',
            'non_violence': DATASET_PATH / 'Hockey' / 'NonViolence',
            'type': 'standard'
        },
        'Movies': {
            'path': DATASET_PATH / 'Movies',
            'violence': DATASET_PATH / 'Movies' / 'Violence',
            'non_violence': DATASET_PATH / 'Movies' / 'NonViolence',
            'type': 'standard'
        },
        'AI4RiSK': {
            'path': DATASET_PATH / 'AI4Risk',
            'non_violence_dirs': ['0'],
            'violence_dirs': ['1', '2', '3', '4'],
            'type': 'multiclass'
        },
    }

    def __init__(self, dataset_name='Crowd'):
        self.set_dataset(dataset_name)
        self.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    def set_dataset(self, dataset_name):
        if dataset_name == 'Mix':
            self.DATASET_NAME = 'Mix'
            self.VIOLENCE_PATH = None
            self.NON_VIOLENCE_PATH = None
            self.SAVE_DIR = Path(f"checkpoints_slowfast_mix")
            self.MODEL_NAME = "slowfast_violence_mix"
        elif dataset_name in self.AVAILABLE_DATASETS:
            self.DATASET_NAME = dataset_name
            dataset_info = self.AVAILABLE_DATASETS[dataset_name]

            if dataset_info.get('type') == 'multiclass':
                self.VIOLENCE_PATH = dataset_info
                self.NON_VIOLENCE_PATH = dataset_info
            else:
                self.VIOLENCE_PATH = dataset_info['violence']
                self.NON_VIOLENCE_PATH = dataset_info['non_violence']

            self.SAVE_DIR = Path(f"checkpoints_slowfast_{dataset_name.lower()}")
            self.MODEL_NAME = f"slowfast_violence_{dataset_name.lower()}"

            if dataset_name == 'AI4RiSK':
                self.SLOWFAST_TAU_FAST = 1
                self.SLOWFAST_TAU_SLOW = 4
                self.NUM_FRAMES_FAST = 32
                self.NUM_FRAMES_SLOW = self.NUM_FRAMES_FAST // self.SLOWFAST_ALPHA
                self.USE_CROP = False
                print(f"\n{'=' * 60}")
                print("AI4RiSK SPECIFIC OPTIMIZATIONS")
                print(f"{'=' * 60}")
                print(f"Issue 1: Short videos (2-4s, 50-100 frames)")
                print(f"  Solution: TAU_FAST = {self.SLOWFAST_TAU_FAST} (dense sampling)")
                print(f"  Temporal window: {self.NUM_FRAMES_FAST} frames (~1.3s at 25fps)")
                print(f"  No frame skipping - captures rapid actions (shooting, etc.)")
                print(f"\nIssue 2: Small subjects, actions at edges (320x240)")
                print(f"  Solution: USE_CROP = {self.USE_CROP}")
                print(f"  Direct resize to 224x224 (no crop)")
                print(f"  Preserves entire frame - no action loss at edges")
                print(f"{'=' * 60}\n")
        else:
            raise ValueError(
                f"Dataset {dataset_name} not found. Available: {list(self.AVAILABLE_DATASETS.keys()) + ['Mix']}")

    def get_mix_paths(self, datasets=None, include_ai4risk=False):
        violence_paths = []
        non_violence_paths = []

        if datasets is None:
            datasets = ['Crowd', 'Hockey', 'Movies']
            if include_ai4risk:
                datasets.append('AI4RiSK')

        for dataset_name in datasets:
            if dataset_name in self.AVAILABLE_DATASETS:
                dataset_info = self.AVAILABLE_DATASETS[dataset_name]

                if dataset_info.get('type') == 'multiclass':
                    violence_paths.append(dataset_info)
                    non_violence_paths.append(dataset_info)
                else:
                    violence_paths.append(dataset_info['violence'])
                    non_violence_paths.append(dataset_info['non_violence'])

        return violence_paths, non_violence_paths