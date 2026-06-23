from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypeAlias

DatasetInfo: TypeAlias = dict[str, Path | list[str] | str]
DatasetPath: TypeAlias = Path | DatasetInfo | list[Path | DatasetInfo] | None


class R3DTransferConfig:
    """Store dataset, model, optimizer, scheduler, and export configuration values."""

    DATASET_PATH: Path = Path("../../Datasets")
    DATASET_NAME: str = "Mix"

    VIOLENCE_PATH: DatasetPath = DATASET_PATH / DATASET_NAME / "Violence"
    NON_VIOLENCE_PATH: DatasetPath = DATASET_PATH / DATASET_NAME / "NonViolence"

    SPLIT_RATIO: float = 0.8
    N_FRAMES: int = 16

    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 100

    OPTIMIZER: str = "adamw"
    BACKBONE_LR: float = 1e-5
    HEAD_LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    BETAS: tuple[float, float] = (0.9, 0.999)
    EPS: float = 1e-8

    FREEZE_LAYERS: list[str] = ["stem", "layer1", "layer2"]
    UNFREEZE_EPOCH: int = 20

    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True

    EARLY_STOPPING_PATIENCE: int = 15

    DROPOUT_P: float = 0.5
    LABEL_SMOOTHING: float = 0.1
    GRAD_CLIP: float = 1.0

    SEED: int = 42

    USE_SCHEDULER: bool = True
    SCHEDULER_TYPE: str = "cosine"
    T_0: int = 10
    T_MULT: int = 2
    ETA_MIN: float = 1e-7

    SAVE_DIR: Path = Path("checkpoints_r3d18_mix")
    MODEL_NAME: str = "r3d18_violence"

    DEVICE: str = "cuda"

    KINETICS_MEAN: list[float] = [0.43216, 0.394666, 0.37645]
    KINETICS_STD: list[float] = [0.22803, 0.22145, 0.216989]

    USE_PRETRAINED: bool = True

    AVAILABLE_DATASETS: dict[str, DatasetInfo] = {
        "Crowd": {
            "path": DATASET_PATH / "Crowd",
            "violence": DATASET_PATH / "Crowd" / "Violence",
            "non_violence": DATASET_PATH / "Crowd" / "NonViolence",
            "type": "standard",
        },
        "Hockey": {
            "path": DATASET_PATH / "Hockey",
            "violence": DATASET_PATH / "Hockey" / "Violence",
            "non_violence": DATASET_PATH / "Hockey" / "NonViolence",
            "type": "standard",
        },
        "Movies": {
            "path": DATASET_PATH / "Movies",
            "violence": DATASET_PATH / "Movies" / "Violence",
            "non_violence": DATASET_PATH / "Movies" / "NonViolence",
            "type": "standard",
        },
        "AI4RiSK": {
            "path": DATASET_PATH / "AI4RISK_CROPPED_SR_V2",
            "non_violence_dirs": ["0"],
            "violence_dirs": ["1", "2", "3", "4"],
            "type": "multiclass",
        },
    }

    def __init__(self, dataset_name: str = "Crowd") -> None:
        """Initialize the configuration for the selected dataset."""
        self.set_dataset(dataset_name)
        self.SAVE_DIR.mkdir(exist_ok=True)

    def set_dataset(self, dataset_name: str) -> None:
        """Update dataset-dependent paths and checkpoint names."""
        if dataset_name == "Mix":
            self.DATASET_NAME = "Mix"
            self.VIOLENCE_PATH = None
            self.NON_VIOLENCE_PATH = None
            self.SAVE_DIR = Path("checkpoints_r3d18_mix")
            self.MODEL_NAME = "r3d18_violence_mix"
            return

        if dataset_name not in self.AVAILABLE_DATASETS:
            available_datasets: list[str] = list(self.AVAILABLE_DATASETS.keys()) + ["Mix"]
            raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {available_datasets}")

        self.DATASET_NAME = dataset_name
        dataset_info: DatasetInfo = self.AVAILABLE_DATASETS[dataset_name]

        if dataset_info.get("type") == "multiclass":
            self.VIOLENCE_PATH = dataset_info
            self.NON_VIOLENCE_PATH = dataset_info
        else:
            self.VIOLENCE_PATH = dataset_info["violence"]
            self.NON_VIOLENCE_PATH = dataset_info["non_violence"]

        self.SAVE_DIR = Path(f"checkpoints_r3d18_{dataset_name.lower()}")
        self.MODEL_NAME = f"r3d18_violence_{dataset_name.lower()}"

    def get_mix_paths(self, datasets: Sequence[str] | None = None) -> tuple[list[Path | DatasetInfo], list[Path | DatasetInfo]]:
        """Return violence and non-violence source paths for the mixed dataset setup."""
        violence_paths: list[Path | DatasetInfo] = []
        non_violence_paths: list[Path | DatasetInfo] = []
        selected_datasets: Sequence[str] = datasets or ["Crowd", "Hockey", "Movies"]

        for dataset_name in selected_datasets:
            if dataset_name not in self.AVAILABLE_DATASETS:
                continue

            dataset_info: DatasetInfo = self.AVAILABLE_DATASETS[dataset_name]
            if dataset_info.get("type") == "multiclass":
                violence_paths.append(dataset_info)
                non_violence_paths.append(dataset_info)
            else:
                violence_paths.append(dataset_info["violence"])
                non_violence_paths.append(dataset_info["non_violence"])

        return violence_paths, non_violence_paths