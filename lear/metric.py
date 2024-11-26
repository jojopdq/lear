import dataclasses


@dataclasses.dataclass
class Metric:
    name: str
    score: float
    category: str
