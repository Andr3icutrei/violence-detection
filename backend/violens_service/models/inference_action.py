from sqlalchemy import BigInteger, Integer, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from core.database import Base
from models.action import Action

class ActionIntType(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, Action):
            return int(value.value)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        raise ValueError(f"Invalid Action value: {value}")

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return Action(int(value))


class InferenceAction(Base):
    __tablename__ = "inference_actions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    action_id: Mapped[Action] = mapped_column(ActionIntType(), nullable=False, unique=True)
    credits: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("action_id <> 0", name="inference_actions_action_id_check"),
    )