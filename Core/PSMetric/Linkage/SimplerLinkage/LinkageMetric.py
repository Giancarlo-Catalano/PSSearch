from typing import Optional

import numpy as np

from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Metric import Metric
from Core.SearchSpace import SearchSpace


class LocalBivariateLinkageMetric(Metric):
    linkage_table: Optional[np.ndarray]
    original_pRef: Optional[PRef]
    local_pRef: Optional[PRef]
    solution: Optional[EvaluatedFS]

    def __init__(self):
        super().__init__()
        self.linkage_table = None
        self.original_pRef = None
        self.local_pRef = None

    def get_atomicity(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()
        if len(fixed_vars) >= 1:
            return self.linkage_table[fixed_vars][fixed_vars]
        else:
            return 0

    def get_single_score(self, ps: PS) -> float:
        return self.get_atomicity(ps)

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

    def set_solution(self, solution: EvaluatedFS):
        # this is where the heavy calculation goes
        self.solution = solution
        self.local_pRef = LocalBivariateLinkageMetric.get_local_pRef(self.original_pRef, self.solution)
        self.linkage_table = self.calculate_linkage_table()

    @classmethod
    def get_local_pRef(cls, original_pRef: PRef, solution: FullSolution):
        # all solutions are written in terms of [same var value as solution or not?]
        new_search_space = SearchSpace([2 for variable in original_pRef.search_space.cardinalities])
        full_solution_matrix = np.array(original_pRef.full_solution_matrix == solution.values, dtype=int)
        return PRef(fitness_array=original_pRef.fitness_array,
                    full_solution_matrix = full_solution_matrix,
                    search_space=new_search_space)


