from typing import Optional

from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Metric import Metric


class SampleCount(Metric):
    pRef: Optional[PRef]

    def __init__(self):
        self.pRef = None
        super().__init__()

    def __repr__(self):
        return "SampleCount"

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

    def get_single_score(self, ps: PS) -> float:
        return len(self.pRef.fitnesses_of_observations(ps))


