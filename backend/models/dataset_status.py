from enum import Enum


class DatasetStatus(Enum):
    INVALID = 0
    PENDING = 10
    ACCEPTED = 20
    REJECTED = 30