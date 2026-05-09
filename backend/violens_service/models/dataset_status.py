from enum import Enum


class DatasetStatus(int, Enum):
    INVALID = 0
    PENDING = 10
    ACCEPTED = 20
    REJECTED = 30