from datetime import datetime
from typing import List

from sqlalchemy import DateTime, func, ForeignKey, Boolean, TypeDecorator, Integer
from sqlalchemy.orm import mapped_column, Mapped, relationship

from core.database import Base
from .inference_model import InferenceModel
from .video import Video
from .user import User

class InferenceModelIntType(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, InferenceModel):
            return int(value.value)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        raise ValueError(f"Invalid Action value: {value}")

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return InferenceModel(int(value))

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    is_official: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    inference_model_id: Mapped[InferenceModel] = mapped_column(InferenceModelIntType(), nullable=False, unique=True)

    created_by_user_id : Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_by_user: Mapped["User"] = relationship(back_populates="datasets_created")

    videos: Mapped[List["Video"]] = relationship(back_populates="dataset")