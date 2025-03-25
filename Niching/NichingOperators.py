import numpy as np

from Core.PRef import PRef
from Core.PS import STAR, PS
from Niching.NichingOperatorInterface import PyMooCustomCrowding


class AverageToeSteppingNiching(PyMooCustomCrowding):
    pRef: PRef

    def __init__(self,
                 pRef: PRef,
                 nds=None, filter_out_duplicates: bool = True):
        self.pRef = pRef
        super().__init__(nds, filter_out_duplicates)

    def get_average_coverage_count(self, pss: list[PS]) -> list[float]:
        counts_per_session = np.zeros(self.pRef.sample_size)
        matching_indexes = [self.pRef.get_indexes_matching_ps(ps) for ps in pss]
        for mi in matching_indexes:
            counts_per_session[mi] += 1

        return [np.average(counts_per_session[mi]) for mi in matching_indexes]

    def get_crowding_scores_of_front(self, all_F, n_remove, population, front_indexes) -> np.ndarray:
        # note that low is good
        pop_matrix = np.array([ind.X for ind in population[front_indexes]])

        def row_to_ps(row):
            return PS(1 if value else STAR for value in row)

        pss = [row_to_ps(row) for row in pop_matrix]
        average_overlaps = self.get_average_coverage_count(pss)
        return np.array(average_overlaps)
