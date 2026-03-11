from sqlalchemy import String, ForeignKey, Integer
from sqlalchemy.orm import mapped_column, Mapped, relationship
from typing import TYPE_CHECKING, List

from core.database import Base
if TYPE_CHECKING:
    from .inference_history import InferenceHistory

class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    inference_history: Mapped[List["InferenceHistory"]] = relationship(back_populates="video")
