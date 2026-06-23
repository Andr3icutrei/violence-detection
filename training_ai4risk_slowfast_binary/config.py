from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class SlowFastConfig:
    """Store configuration values for the SlowFast violence-detection pipeline."""
    DATASET_PATH: Path = Path('../../Datasets')
    DATASET_NAME: str = 'AI4RiSK_CROPPED_SR_V2'
    SEED: int = 42
    SPLIT_RATIO: float = 0.8
    NUM_CLASSES: int = 2
    CLASS_NAMES: List[str] = ["Non-Violence", "Violence"]

    SLOWFAST_ALPHA: int = 4
    SLOWFAST_BETA: float = 0.125
    SLOW_FRAMES: int = 8
    FAST_FRAMES: int = 32
    TEMPORAL_STRIDE: int = 1

    INPUT_SIZE: int = 224
    CROP_SIZE: int = 224

    USE_CROP: bool = False

    BATCH_SIZE_FROZEN: int = 16
    BATCH_SIZE_UNFROZEN: int = 8

    NUM_EPOCHS: int = 100

    ACCUMULATION_STEPS_FROZEN: int = 2
    ACCUMULATION_STEPS_UNFROZEN: int = 4

    OPTIMIZER: str = 'adamw'
    BACKBONE_LR: float = 1e-05
    HEAD_LR: float = 0.0001
    WEIGHT_DECAY: float = 0.01
    BETAS: Tuple[float, float] = (0.9, 0.999)
    EPS: float = 1e-08

    FREEZE_BACKBONE: bool = True
    UNFREEZE_EPOCH: int = 20

    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    EARLY_STOPPING_PATIENCE: int = 15
    DROPOUT_P: float = 0.5
    LABEL_SMOOTHING: float = 0.1
    GRAD_CLIP: float = 5.0
    USE_CLASS_WEIGHTS: bool = True
    USE_BALANCED_SAMPLING: bool = False
    USE_AMP: bool = True
    USE_SCHEDULER: bool = True
    SCHEDULER_TYPE: str = 'cosine'
    T_0: int = 10
    T_MULT: int = 2
    ETA_MIN: float = 1e-07
    SAVE_DIR: Path = Path('checkpoints_slowfast_ai4risk_binary')
    MODEL_NAME: str = 'slowfast_violence_ai4risk_binary'
    DEVICE: str = 'cuda'
    KINETICS_MEAN: List[float] = [0.45, 0.45, 0.45]
    KINETICS_STD: List[float] = [0.225, 0.225, 0.225]
    USE_PRETRAINED: bool = True
    AVAILABLE_DATASETS: Dict[str, dict] = {'AI4RiSK': {'path': DATASET_PATH / 'AI4RiSK_CROPPED_SR_V2', 'non_violence_dirs': ['0'], 'violence_dirs': ['1', '2', '3', '4'], 'type': 'multiclass'}}
    VIOLENCE_PATH: dict
    NON_VIOLENCE_PATH: dict

    def __init__(self) -> None:
        """Initialize the object and its runtime state."""
        self.set_dataset('AI4RiSK')
        self.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    def set_dataset(self, dataset_name: str) -> None:
        """Select the dataset metadata used by the configuration."""
        if dataset_name == 'AI4RiSK':
            self.DATASET_NAME: str = 'AI4RiSK'
            dataset_info: dict = self.AVAILABLE_DATASETS['AI4RiSK']
            self.VIOLENCE_PATH = dataset_info
            self.NON_VIOLENCE_PATH = dataset_info
            self.SAVE_DIR = Path('checkpoints_slowfast_ai4risk_binary')
            self.MODEL_NAME: str = 'slowfast_violence_ai4risk_binary'
            self.USE_CROP: bool = False
        else:
            raise ValueError(f'Dataset {dataset_name} not supported. Only AI4RiSK is available.')

    @property
    def BATCH_SIZE(self) -> int:
        """Return the active batch size for the current training phase."""
        return self.BATCH_SIZE_FROZEN if self.FREEZE_BACKBONE else self.BATCH_SIZE_UNFROZEN

    @property
    def ACCUMULATION_STEPS(self) -> int:
        """Return the active gradient-accumulation step count."""
        return self.ACCUMULATION_STEPS_FROZEN if self.FREEZE_BACKBONE else self.ACCUMULATION_STEPS_UNFROZEN

    @property
    def EFFECTIVE_BATCH_SIZE(self) -> int:
        """Return the batch size after gradient accumulation."""
        return self.BATCH_SIZE * self.ACCUMULATION_STEPS