import asyncio
import cv2
from pathlib import Path
from inference_service.core import SessionLocal
from models.dataset import Dataset
from models.video import Video

DATASETS_PATH = Path("../../../../Datasets")

available_dataset = {
    "AI4RiSK": {
        "path": DATASETS_PATH / "AI4RiSK",
        "non_violence_dirs": ['0'],
        "violence_dirs": ['1', '2', '3', '4'],
        "dataset_id": 1
    },
    "Movies": {
        "path": DATASETS_PATH / "Movies",
        "violence_dirs": ['Violence'],
        "non_violence_dirs": ['NonViolence'],
        "dataset_id": 2
    },
    "Hockey": {
        "path": DATASETS_PATH / "Hockey",
        "violence_dirs": ['Violence'],
        "non_violence_dirs": ['NonViolence'],
        "dataset_id": 3
    },
    "Crowd": {
        "path": DATASETS_PATH / "Crowd",
        "violence_dirs": ['Violence'],
        "non_violence_dirs": ['NonViolence'],
        "dataset_id": 4
    },
}

async def insert_videos():
    async with SessionLocal() as db:
        for dataset_key, dataset_value in available_dataset.items():
            try:
                for dir in dataset_value["violence_dirs"] + dataset_value["non_violence_dirs"]:
                    dir_path = dataset_value["path"] / dir
                    for video_file in dir_path.glob("*"):
                        cap = cv2.VideoCapture(str(video_file))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration_sec = frame_count / fps if fps > 0 else 0
                        cap.release()

                        video = Video(
                            name=video_file.name,
                            path=str(Path(dataset_key) / Path(dir) / Path(video_file.name)).replace("\\", "/"),
                            dataset_id=dataset_value["dataset_id"],
                            is_violent= True if dir in dataset_value["violence_dirs"] else False,
                            duration=duration_sec,
                            frame_rate=fps
                        )
                        db.add(video)
                await db.commit()
            except Exception as e:
                print(f"Error inserting videos for dataset {dataset_key}: {e}")
                await db.rollback()

async def main() -> None:
    await insert_videos()

if __name__ == "__main__":
    asyncio.run(main())