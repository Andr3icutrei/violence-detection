from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

from models.dataset import Dataset
from models.video import Video

DATASETS_PATH = Path("../../../../Datasets")

available_dataset = {
    "AI4RiSK": {
        "path": DATASETS_PATH / "AI4RiSK",
        "violence_dirs": ['0'],
        "non_violence_dirs": ['1', '2', '3', '4'],
        "dataset_id": Dataset.AI4RiSK
    },
    "Movies": {
        "path": DATASETS_PATH / "Movies",
        "violence_dirs": ['Violence'],
        "non_violence_dirs": ['NonViolence'],
        "dataset_id": Dataset.MOVIES
    },
    "Hockey": {
        "path": DATASETS_PATH / "Hockey",
        "violence_dirs": ['Violence'],
        "non_violence_dirs": ['NonViolence'],
        "dataset_id": Dataset.HOCKEY
    },
    "Crowd": {
        "path": DATASETS_PATH / "Crowd",
        "violence_dirs": ['Violence'],
        "non_violence_dirs": ['NonViolence'],
        "dataset_id": Dataset.CROWD
    },
}

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable is not set")

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def insert_videos():
    db = SessionLocal()

    for dataset_key, dataset_value in available_dataset.items():
        try:
            for dir in dataset_value["violence_dirs"] + dataset_value["non_violence_dirs"]:
                dir_path = dataset_value["path"] / dir
                for video_file in dir_path.glob("*"):
                    video = Video(name=video_file.name, path=str(Path(dataset_key) / Path(dir) / Path(video_file.name)).replace("\\", "/"), dataset_id=dataset_value["dataset_id"].value)
                    db.add(video)
            db.commit()
        except Exception as e:
            print(f"Error inserting videos for dataset {dataset_key}: {e}")
            db.rollback()
        finally:
            db.close()

if __name__ == "__main__":
    insert_videos()