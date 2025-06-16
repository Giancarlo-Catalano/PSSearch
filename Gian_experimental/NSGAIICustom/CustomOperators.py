import random

import numpy as np

from Core.PRef import PRef
from Gian_experimental.NSGAIICustom.NSGAIICustom import NCSolution, NCCrossover, sample_from_probabilities, NCMutation, \
    NCSampler
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import scale_to_have_sum


def hot_encode_and_multiply(solution: NCSolution, transition_matrix: np.ndarray) -> np.ndarray:
    hot_encoded = np.zeros(transition_matrix.shape[1], dtype=float)
    hot_encoded[list(solution)] = 1
    return hot_encoded.reshape((1, -1)) @ transition_matrix


class NCSamplerFromPRef(NCSampler):
    probabilities_of_existing: np.ndarray
    allow_empty: bool

    def __init__(self, probabilities_of_existing: np.ndarray, allow_empty: bool):
        super().__init__()
        self.probabilities_of_existing = probabilities_of_existing
        self.probabilities_of_existing = scale_to_have_sum(self.probabilities_of_existing, wanted_sum=4)
        self.allow_empty = allow_empty

    def sample(self) -> NCSolution:
        produced = sample_from_probabilities(self.probabilities_of_existing)
        if len(produced) == 0 and not self.allow_empty:
            produced.add(random.choice(range(self.probabilities_of_existing)))
        return produced

    @classmethod
    def from_PRef(cls, pRef: PRef, allow_empty: bool = False):
        # assumes that pRef is boolean
        probabilities = np.average(pRef.full_solution_matrix, axis=0)
        return cls(probabilities, allow_empty=allow_empty)


class NCMutationCounterproductive(NCMutation):
    transition_matrix: np.ndarray
    n: int

    def __init__(self, transition_probabilities):
        super().__init__()
        self.transition_matrix = transition_probabilities
        self.n = transition_probabilities.shape[1]

    def mutated(self, solution: NCSolution) -> NCSolution:
        probabilities = hot_encode_and_multiply(solution, self.transition_matrix)
        probabilities = probabilities.ravel()
        disappearance_probability = 1 / len(solution) if len(solution) > 0 else 0  # TODO document this
        # probabilities = scale_to_have_sum_and_max(probabilities,
        #                                           wanted_sum=len(solution),
        #                                           wanted_max=1 - disappearance_probability,
        #                                           positions=self.n)
        probabilities = scale_to_have_sum(probabilities, len(solution))  # it should already be like that...
        # print(probabilities)
        return sample_from_probabilities(probabilities)


class NCCrossoverTransition(NCCrossover):
    transition_matrix: np.ndarray

    def __init__(self, transition_probabilities):
        super().__init__()
        self.transition_matrix = transition_probabilities

    def crossed(self, a: NCSolution, b: NCSolution):
        guaranteed = a.intersection(b)
        considering = a ^ b

        considering_probabilities = hot_encode_and_multiply(considering, self.transition_matrix)
        considering_probabilities = considering_probabilities.ravel()
        wanted_quantity_of_ones = ((len(a) + len(b)) / 2) - len(guaranteed)
        # considering_probabilities = scale_to_have_sum_and_max(considering_probabilities,
        #                                                       wanted_sum=wanted_quantity_of_ones,
        #                                                       wanted_max=1,
        #                                                       positions=len(considering))
        considering_probabilities = scale_to_have_sum(considering_probabilities,
                                                      wanted_sum=wanted_quantity_of_ones)
        child_1 = guaranteed.copy()
        child_2 = guaranteed.copy()

        for considered_index in considering:
            if random.random() < considering_probabilities[considered_index]:
                child_1.add(considered_index)
            else:
                child_2.add(considered_index)

        return child_1, child_2
