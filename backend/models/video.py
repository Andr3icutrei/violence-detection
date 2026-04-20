import uuid
from datetime import datetime

from sqlalchemy import String, ForeignKey, Integer, Boolean, Uuid, DateTime
from sqlalchemy.orm import mapped_column, Mapped, relationship
from typing import TYPE_CHECKING, List

from core.database import Base
if TYPE_CHECKING:
    from .inference_history import InferenceHistory
    from .dataset import Dataset

class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    uid: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    is_violent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    duration: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    frame_rate: Mapped[float] = mapped_column(Integer, nullable=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    dataset: Mapped["Dataset"] = relationship(back_populates="videos")

    inference_history: Mapped[List["InferenceHistory"]] = relationship(back_populates="video")
