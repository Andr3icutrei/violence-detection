from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class InferenceHistoryClassification(Base):
    __tablename__ = "inference_history_classification"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    ground_truth: Mapped[int] = mapped_column(nullable=False)
    prediction: Mapped[int] = mapped_column(nullable=False)

    inference_history_id: Mapped[int] = mapped_column(ForeignKey("inference_history.id"), index=True, nullable=False)
    inference_history: Mapped["InferenceHistory"] = relationship(back_populates="inference_history_classification")