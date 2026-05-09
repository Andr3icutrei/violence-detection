from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, func, Integer, CheckConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List, TYPE_CHECKING

from core.database import Base

if TYPE_CHECKING:
    from .inference_history import InferenceHistory
    from .dataset import Dataset

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    credits: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_account_verified: Mapped[bool] = mapped_column(Boolean, nullable=False)
    auth_provider: Mapped[str] = mapped_column(String(50), nullable=True)
    is_banned: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    ban_reason: Mapped[str] = mapped_column(String(255), nullable=True)

    inference_history: Mapped[List["InferenceHistory"]] = relationship(back_populates="user")

    datasets_created: Mapped[List["Dataset"]] = relationship(back_populates="created_by_user")

    __table_args__ = (
        CheckConstraint("credits >= 0", name="users_credits_check"),
    )