from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
import os
import tempfile

from models.action import Action
from models.dataset_status import DatasetStatus


def make_user(*, user_id: int = 1, is_admin: bool = False):
    return SimpleNamespace(
        id=user_id,
        email="user@example.com",
        credits=10,
        is_active=True,
        is_admin=is_admin,
        is_banned=False,
        ban_reason=None,
        credits_used=0,
        is_account_verified=True,
        hashed_password="hashed",
        auth_provider="local",
    )


def make_user_response(*, user_id: int = 1, is_admin: bool = False):
    user = make_user(user_id=user_id, is_admin=is_admin)
    return SimpleNamespace(
        id=user.id,
        email=user.email,
        credits=user.credits,
        is_active=user.is_active,
        is_admin=user.is_admin,
        is_banned=user.is_banned,
        ban_reason=user.ban_reason,
        credits_used=user.credits_used,
    )


def make_dataset(*, dataset_id: int = 1, status: DatasetStatus = DatasetStatus.ACCEPTED):
    return SimpleNamespace(
        id=dataset_id,
        name="Sample Dataset",
        is_official=False,
        status=status,
    )


def make_video(*, video_id: int = 1, dataset_name: str = "Sample Dataset"):
    dataset = SimpleNamespace(name=dataset_name, is_official=False)
    return SimpleNamespace(
        id=video_id,
        uid="video-uid",
        name="video.mp4",
        path="/tmp/video.mp4",
        is_violent=False,
        dataset_id=1,
        dataset=dataset,
        duration=120,
        frame_rate=30.0,
    )


@dataclass
class InferenceResult:
    video_path: str
    predicted_label: str
    predicted_class_probability: float
    confidence: float


class FakeAuthService:
    async def login(self, email: str, password: str):
        return make_user()

    async def login_google(self, data):
        return make_user()


class FakeUsersService:
    async def create_user(self, user_create_data, conf):
        return make_user()

    async def verify_account(self, token: str):
        return make_user()

    async def reset_password(self, token: str, new_password: str):
        return make_user()

    async def send_reset_password_email(self, email: str, conf):
        return None

    async def verify_reset_password_token(self, token: str):
        return {"valid": True}

    async def resend_verification_email(self, token: str, conf):
        return make_user()

    async def get_user_by_id(self, user_id: int):
        return make_user(user_id=user_id)

    async def get_all_users(self, search_term, page: int, page_size: int):
        return [make_user(user_id=1), make_user(user_id=2)]

    async def update_user_role(self, user_id: int, is_admin: bool):
        return None

    async def ban_user(self, user_id: int, ban_reason: str, conf):
        return None

    async def get_users_stats(self):
        return {
            "active_users": 3,
            "inactive_users": 1,
            "banned_users": 0,
            "most_active_users": [make_user(user_id=1)],
        }

    async def update_all_users_credits(self):
        return None


class FakeDatasetsService:
    async def get_accepted_datasets(self):
        return [make_dataset(dataset_id=1)]

    async def create_unofficial_dataset(self, create_dataset_dto, user_id: int):
        return None

    async def get_datasets(self, search_term, page: int, page_size: int, dataset_status, is_official):
        dataset = make_dataset(dataset_id=2, status=DatasetStatus.PENDING)
        return [
            SimpleNamespace(
                **dataset.__dict__,
                user=make_user_response(user_id=1),
                videos_count=2,
                violent_videos_count=1,
                non_violent_videos_count=1,
            )
        ]

    async def get_dataset_videos(self, dataset_id: int):
        dataset = make_dataset(dataset_id=dataset_id)
        return SimpleNamespace(
            **dataset.__dict__,
            videos=[
                {
                    "id": 1,
                    "uid": "video-uid",
                    "name": "video.mp4",
                    "path": "/tmp/video.mp4",
                    "dataset_id": dataset_id,
                    "dataset_name": dataset.name,
                    "dataset_is_official": dataset.is_official,
                    "is_violent": False,
                    "duration": 120,
                    "frame_rate": 30,
                }
            ],
            inference_model_name="model.onnx",
            inference_model_path="/models/model.onnx",
        )

    async def review_dataset(self, dataset_id: int, is_approved: bool, videos, review_comment: str, conf, excluded_video_ids):
        return make_dataset(dataset_id=dataset_id, status=DatasetStatus.ACCEPTED)

    async def delete_dataset(self, dataset_id: int):
        return None

    async def edit_dataset(self, dataset_id: int, videos, excluded_video_ids):
        return make_dataset(dataset_id=dataset_id)

    async def validate_dataset_model(self, dataset_id: int, videos, excluded_video_ids):
        return {
            "accuracy": 0.9,
            "confusion_matrix": {
                "true_positive": 8,
                "true_negative": 7,
                "false_positive": 1,
                "false_negative": 2,
            },
        }

    async def get_datasets_stats(self):
        return {
            "most_popular_dataset_classification": {
                "id": 1,
                "name": "Dataset A",
                "is_official": True,
                "status": DatasetStatus.ACCEPTED,
                "inferences_videos_count": 10,
            },
            "most_popular_dataset_people_tracking": {
                "id": 2,
                "name": "Dataset B",
                "is_official": False,
                "status": DatasetStatus.ACCEPTED,
                "inferences_videos_count": 7,
            },
            "official_datasets_count": 2,
            "unofficial_datasets_count": 1,
            "pending_datasets_count": 0,
            "storage_used_gb": 12.5,
        }


class FakeCreditsService:
    async def get_credits_cronjob_update(self):
        return {"default_credits": 10}

    async def patch_credits_cronjob_update(self, new_credits: int):
        return None


class FakeInferenceActionsService:
    async def get_inference_actions_for_dataset(self, dataset_id: int):
        return [SimpleNamespace(id=1, action_id=Action.CLASSIFICATION, credits=5)]

    async def get_inference_actions_stats(self):
        return [SimpleNamespace(id=2, action_id=Action.PEOPLE_TRACKING, credits=3)]

    async def update_credits_for_action(self, actions_to_patch):
        return None


class FakeInferenceHistoryService:
    async def get_inference_history_stats(self, year: int, month: int):
        now = datetime.now(timezone.utc)
        return {
            "classification_runs": [
                {
                    "id": 1,
                    "ground_truth": True,
                    "prediction": True,
                    "created_at": now,
                }
            ],
            "people_tracking_runs": [
                {
                    "id": 2,
                    "people_tracked": 4,
                    "created_at": now,
                }
            ],
        }


class FakeVideosService:
    def __init__(self, temp_dir: str | None = None):
        self.temp_dir = temp_dir
        self.last_temp_path: str | None = None

    async def get_videos_paged(self, search_term, dataset_id, is_violent, dataset_status, asc, page, page_size):
        return [make_video(video_id=1)]

    async def exists_video(self, video_uid: str) -> bool:
        return video_uid == "exists"

    async def classify_and_occlusion_video(self, video_id: int, current_user):
        video_path = self._write_temp_video("gradcam")
        return InferenceResult(
            video_path=video_path,
            predicted_label="violent",
            predicted_class_probability=0.82,
            confidence=0.91,
        )

    async def people_tracking(self, video_id: int, current_user):
        video_path = self._write_temp_video("tracking")
        return video_path, 3

    def _write_temp_video(self, prefix: str) -> str:
        directory = self.temp_dir or tempfile.gettempdir()
        os.makedirs(directory, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".mp4", dir=directory)
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"data")
        self.last_temp_path = path
        return path
