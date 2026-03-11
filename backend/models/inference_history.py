from datetime import datetime
from sqlalchemy import DateTime, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import TYPE_CHECKING

from core.database import Base
if TYPE_CHECKING:
    from .video import Video
    from .user import User

class InferenceHistory(Base):
    __tablename__ = "inference_history"

    id:Mapped[int] = mapped_column(primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ground_truth: Mapped[str] = mapped_column(nullable=True)
    prediction: Mapped[str] = mapped_column(nullable=True)

    video_id:Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    user_id:Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    video: Mapped["Video"] = relationship(back_populates="inference_history")
    user: Mapped["User"] = relationship(back_populates="inference_history")