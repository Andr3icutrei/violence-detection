from core.database import Base
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .dataset import Dataset

class InferenceModel(Base):
    __tablename__ = "inference_models"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    path: Mapped[str] = mapped_column(String(255), nullable=False)

    datasets: Mapped[List["Dataset"]] = relationship(back_populates="inference_model")
