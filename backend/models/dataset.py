from datetime import datetime
from typing import List

from sqlalchemy import DateTime, func, ForeignKey, Boolean
from sqlalchemy.orm import mapped_column, Mapped, relationship

from core.database import Base
from .video import Video
from .user import User

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    is_official: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    created_by_user_id : Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_by_user: Mapped["User"] = relationship(back_populates="datasets_created")

    videos: Mapped[List["Video"]] = relationship(back_populates="dataset")