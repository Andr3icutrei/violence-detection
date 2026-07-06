from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class MViTConfig:
    """Store configuration values for the MViT violence-detection pipeline."""
    DATASET_PATH: Path = Path('../../Datasets')
    DATASET_NAME: str = 'AI4RiSK_CROPPED_SR_V2'
    SEED: int = 42
    SPLIT_RATIO: float = 0.8
    NUM_CLASSES: int = 2
    CLASS_NAMES: List[str] = ["Non-Violence", "Violence"]

    BATCH_SIZE: int = 4

    NUM_FRAMES: int = 16
    TEMPORAL_STRIDE: int = 3

    INPUT_SIZE: int = 224
    CROP_SIZE: int = 224

    USE_CROP: bool = False

    NUM_EPOCHS: int = 100

    ACCUMULATION_STEPS: int = 8

    OPTIMIZER: str = 'adamw'
    BACKBONE_LR: float = 1e-05
    HEAD_LR: float = 0.0001
    WEIGHT_DECAY: float = 0.05
    BETAS: Tuple[float, float] = (0.9, 0.999)
    EPS: float = 1e-08

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

    SAVE_DIR: Path = Path('checkpoints_mvit_ai4risk_binary')
    MODEL_NAME: str = 'mvit_violence_ai4risk_binary'
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
            self.SAVE_DIR = Path('checkpoints_mvit_ai4risk_binary')
            self.MODEL_NAME: str = 'mvit_violence_ai4risk_binary'
            self.USE_CROP: bool = False
        else:
            raise ValueError(f'Dataset {dataset_name} not supported. Only AI4RiSK is available.')

    @property
    def EFFECTIVE_BATCH_SIZE(self) -> int:
        """Return the batch size after gradient accumulation."""
        return self.BATCH_SIZE * self.ACCUMULATION_STEPS