from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional


DatasetInfo = dict[str, Path | list[str] | str]


class X3DConfig:
    """Stores dataset, model, optimizer, scheduler, and runtime configuration."""

    DATASET_PATH: ClassVar[Path] = Path("../../Datasets")
    DATASET_NAME: ClassVar[str] = "AI4RiSK_CROPPED_SR_V2"

    VIOLENCE_PATH: Optional[DatasetInfo] = None
    NON_VIOLENCE_PATH: Optional[DatasetInfo] = None

    SPLIT_RATIO: float = 0.8

    X3D_VERSION: str = "m"

    NUM_FRAMES: int = 16
    TEMPORAL_STRIDE: int = 2

    INPUT_SIZE: int = 224
    CROP_SIZE: int = 224

    USE_CROP: bool = False

    BATCH_SIZE: int = 12
    NUM_EPOCHS: int = 100

    ACCUMULATION_STEPS: int = 3
    EFFECTIVE_BATCH_SIZE: int = BATCH_SIZE * ACCUMULATION_STEPS

    OPTIMIZER: str = "adamw"
    BACKBONE_LR: float = 1e-5
    HEAD_LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-3
    BETAS: tuple[float, float] = (0.9, 0.999)
    EPS: float = 1e-8

    FREEZE_BACKBONE: bool = False
    UNFREEZE_EPOCH: int = 20

    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True

    EARLY_STOPPING_PATIENCE: int = 15

    DROPOUT_P: float = 0.5
    LABEL_SMOOTHING: float = 0.1
    GRAD_CLIP: float = 2.0

    USE_SCHEDULER: bool = True
    SCHEDULER_TYPE: str = "cosine"
    T_0: int = 10
    T_MULT: int = 2
    ETA_MIN: float = 1e-7

    SAVE_DIR: Path = Path("checkpoints_x3d_ai4risk")
    MODEL_NAME: str = "x3d_violence_ai4risk"

    DEVICE: str = "cuda"

    KINETICS_MEAN: list[float] = [0.45, 0.45, 0.45]
    KINETICS_STD: list[float] = [0.225, 0.225, 0.225]

    USE_PRETRAINED: bool = True
    USE_OPTICAL_FLOW: bool = True

    AVAILABLE_DATASETS: ClassVar[dict[str, DatasetInfo]] = {
        "AI4RiSK": {
            "path": DATASET_PATH / "AI4RiSK_CROPPED_SR_V2",
            "non_violence_dirs": ["0"],
            "violence_dirs": ["1", "2", "3", "4"],
            "type": "multiclass",
        },
    }

    def __init__(self) -> None:
        """Initializes the default dataset configuration and checkpoint directory."""

        self.set_dataset("AI4RiSK")
        self.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    def set_dataset(self, dataset_name: str) -> None:
        """Selects a supported dataset and updates dataset-specific paths."""

        dataset_info: Optional[DatasetInfo] = self.AVAILABLE_DATASETS.get(dataset_name)
        if dataset_info is None:
            supported_datasets: str = ", ".join(self.AVAILABLE_DATASETS)
            raise ValueError(f"Dataset '{dataset_name}' is not supported. Available datasets: {supported_datasets}.")

        self.DATASET_NAME = dataset_name
        self.VIOLENCE_PATH = dataset_info
        self.NON_VIOLENCE_PATH = dataset_info
        self.SAVE_DIR = Path(f"checkpoints_x3d_{dataset_name.lower()}")
        self.MODEL_NAME = f"x3d_violence_{dataset_name.lower()}"
        self.USE_CROP = False