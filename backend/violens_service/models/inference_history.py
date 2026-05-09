from datetime import datetime
from sqlalchemy import DateTime, func, ForeignKey, Integer, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List

from core.database import Base
from .video import Video
from .user import User
from .inference_history_classification import InferenceHistoryClassification
from .inference_history_people_tracking import InferenceHistoryPeopleTracking

class InferenceHistory(Base):
    __tablename__ = "inference_history"

    id:Mapped[int] = mapped_column(primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    credits_used: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

    video_id:Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    user_id:Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    video: Mapped["Video"] = relationship(back_populates="inference_history")
    user: Mapped["User"] = relationship(back_populates="inference_history")

    inference_history_classification: Mapped[List["InferenceHistoryClassification"]] = relationship(back_populates="inference_history")
    inference_history_people_tracking: Mapped[List["InferenceHistoryPeopleTracking"]] = relationship(back_populates="inference_history")