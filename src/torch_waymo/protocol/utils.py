from enum import IntEnum


class ReversibleIntEnum(IntEnum):
    @classmethod
    def as_dict(cls):
        return cls.__members__

    @classmethod
    def reverse_dict(cls):
        return {v: k for k, v in cls.__members__.items()}

    def to_idx(self):
        return self - 1
