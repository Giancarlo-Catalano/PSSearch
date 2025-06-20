import heapq
import itertools
import math
import random
from typing import Optional

import utils
from Core.PRef import PRef
import numpy as np

from Core.PS import PS
from PolishSystem.read_data import get_pRef_from_vectors
from retail_forecasting_data_collection.data_file_names import vector_path, fitness_values_path


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
    quantity_of_sessions: int

    def __init__(self, sessions: list[set[int]],
                 fitnesses: np.ndarray):
        super().__init__(sessions, fitnesses)

        max_product = max(product for session in self.sessions
                          for product in session)
        self.which_sessions = [{index for index, session in enumerate(sessions)
                                if product in session}
                               for product in range(max_product + 1)]

        self.quantity_of_sessions = len(sessions)

    def partition_using_threshold_outdated(self, ps: set[int], threshold: float):
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

    def partition_using_threshold(self, ps: set[int], threshold: int):
        if len(ps) < threshold or len(ps) < 1:
            return np.array([]), self.fitnesses  # shortcut

        by_match_count = [set() for match_count in range(threshold + 1)]

        def consider_var(var_index):
            # by_match_count[0] should be all indices, but it's faster to keep that empty and just add the indices to [1] directly
            sessions_with_that_item = self.which_sessions[var_index]
            for match_count in reversed(range(1, threshold + 1)):
                by_match_count[match_count].update(
                    by_match_count[match_count - 1].intersection(sessions_with_that_item))

            by_match_count[1].update(sessions_with_that_item)

        for var in ps:
            consider_var(var)

        index_matches = by_match_count[-1]
        # Create a mask to get values NOT at those indices
        mask = np.ones(self.quantity_of_sessions, dtype=bool)
        mask[list(index_matches)] = False
        not_selected = self.fitnesses[mask]

        return self.fitnesses[list(index_matches)], not_selected


def check_if_optimisation_works():
    local_vector_path = r"C:\Users\gac8\PycharmProjects\PSSearch\retail_forecasting_data_collection\data\many_hot_vectors_250_random.csv"
    local_fitness_path = r"C:\Users\gac8\PycharmProjects\PSSearch\retail_forecasting_data_collection\data\fitness_250_random.csv"
    original_pRef = get_pRef_from_vectors(name_of_vectors_file=local_vector_path,
                                          name_of_fitness_file=local_fitness_path,
                                          column_in_fitness_file=2)

    optimised_SPref = OptimisedSPref.from_pRef(original_pRef)

    def check_if_splits_are_identical(values_1, values_2):
        return np.array_equal(sorted(values_1.flat), sorted(values_2.flat))

    pss = [{a, b} for a, b in itertools.combinations(range(250), r=2)][::10]
    print(len(pss))

    for ps in pss:
        threshold = 2
        match_traditional, unmatch_traditional = optimised_SPref.partition_using_threshold(ps, threshold)
        # match_new, unmatch_new = optimised_SPref.faster_partition_using_threshold(ps, threshold)

        # if not check_if_splits_are_identical(match_traditional, match_new):
        #     print(f"For PS {ps} at threshold {threshold}, there is a mismatch! between the matches")
        #     print(f"Lengths are {len(match_traditional)}, {len(match_new)}")
        #
        # if not check_if_splits_are_identical(unmatch_traditional, unmatch_new):
        #     print(f"For PS {ps} at threshold {threshold}, there is a mismatch! between the matches")
        #     print(f"Lengths are {len(unmatch_traditional)}, {len(unmatch_new)}")

    print("All finished!")

# with utils.announce("timing things"):
#    check_if_optimisation_works()
