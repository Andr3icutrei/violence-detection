from datetime import datetime, date
from typing import List, TYPE_CHECKING

from sqlalchemy import Date, DateTime, func, ForeignKey, Boolean, TypeDecorator, Integer, String, Text, BigInteger, CheckConstraint, text
from sqlalchemy.orm import mapped_column, Mapped, relationship

from core.database import Base
from .dataset_status import DatasetStatus
from .video import Video
from .user import User

if TYPE_CHECKING:
    from .inference_model import InferenceModel

class DatasetStatusIntType(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, DatasetStatus):
            return int(value.value)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        raise ValueError(f"Invalid Action value: {value}")

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return DatasetStatus(int(value))



class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False, server_default=text("''"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at: Mapped[date] = mapped_column(Date, nullable=True)
    is_official: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    status: Mapped[DatasetStatus] = mapped_column(DatasetStatusIntType(), nullable=False, server_default=text("10"))
    comment: Mapped[str] = mapped_column(Text, nullable=True)

    inference_model_id: Mapped[int] = mapped_column(ForeignKey("inference_models.id"), nullable=True)
    inference_model: Mapped["InferenceModel"] = relationship(back_populates="datasets")

    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_by_user: Mapped["User"] = relationship(back_populates="datasets_created")

    videos: Mapped[List["Video"]] = relationship(back_populates="dataset")


    __table_args__ = (
        CheckConstraint("status > 0", name="datasets_status_check"),
        CheckConstraint("inference_model_id <> 0", name="datasets_inference_model_id_check"),
    )