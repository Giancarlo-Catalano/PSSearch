import heapq
import math
from typing import Optional

import utils
from Core.PRef import PRef
import numpy as np


class SPRef:
    sessions: list[set[int]]
    fitnesses: np.ndarray

    def __init__(self,
                 sessions: list[set[int]],
                 fitnesses: np.ndarray):
        self.sessions = sessions
        self.fitnesses = fitnesses

    @classmethod
    def from_pRef(cls, pRef: PRef):
        def row_to_set(row):
            return {index for index, value in enumerate(row) if value}

        return cls(list(map(row_to_set, pRef.full_solution_matrix)), pRef.fitness_array)

    def partition_using_threshold(self, ps: set[int], threshold: float):
        if len(ps) < threshold:
            return np.array([]), self.fitnesses  # shortcut

        most_leftover = len(ps) - threshold
        matches = []
        non_matches = []

        for session, fitness in zip(self.sessions, self.fitnesses):
            if len(ps - session) <= most_leftover:
                matches.append(fitness)
            else:
                non_matches.append(fitness)
        return np.array(matches), np.array(non_matches)

    def partition(self, ps: set[int], threshold: Optional[float]):
        if threshold is None:
            return self.partition_using_threshold(ps, threshold=len(ps))
        else:
            return self.partition_using_threshold(ps, threshold)


class OptimisedSPref(SPRef):
    which_sessions: list[set[int]]

    def __init__(self, sessions: list[set[int]],
                 fitnesses: np.ndarray):
        super().__init__(sessions, fitnesses)

        max_product = max(product for session in self.sessions
                          for product in session)
        self.which_sessions = [{index for index, session in enumerate(sessions)
                                if product in session}
                               for product in range(max_product + 1)]

    def partition_using_threshold(self, ps: set[int], threshold: float):
        if len(ps) < threshold or len(ps) < 1:
            return np.array([]), self.fitnesses  # shortcut

        most_leftover = len(ps) - threshold
        index_matches = set()
        possible_session_indices = set.union(*(self.which_sessions[product] for product in ps))
        index_non_matches = set(range(len(self.sessions))) - possible_session_indices

        for index in possible_session_indices:
            session = self.sessions[index]
            if len(ps - session) <= most_leftover:
                index_matches.add(index)
            else:
                index_non_matches.add(index)

        return np.array(self.fitnesses[list(index_matches)]), np.array(self.fitnesses[list(index_non_matches)])
