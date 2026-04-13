from sqlalchemy import Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base

class InferenceHistoryPeopleTracking(Base):
    __tablename__ = "inference_history_people_tracking"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    people_tracked: Mapped[int] = mapped_column(Integer, nullable=False)

    inference_history_id: Mapped[int] = mapped_column(ForeignKey("inference_history.id"), index=True, nullable=False)
    inference_history: Mapped["InferenceHistory"] = relationship(back_populates="inference_history_people_tracking")