import random

import numpy as np
from pymoo.core.crossover import Crossover

from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import scale_to_have_sum_and_max, \
    sample_PS_from_probabilties_for_global, from_global_to_zeroone, scale_to_have_sum


class TransitionCrossover(Crossover):
    transition_matrix: np.ndarray
    n: int
    def __init__(self,
                 transition_matrix: np.ndarray,
                 n_offsprings=2,
                 prob: float = 0.5,
                 **kwargs):
        super().__init__(2, n_offsprings, prob = prob, **kwargs)
        self.transition_matrix = transition_matrix
        self.n = transition_matrix.shape[0]

    def ps_uniform_crossover(self, mother: np.ndarray, father: np.ndarray):
        guaranteed = np.logical_and(mother, father)
        considering = np.logical_xor(mother, father)

        probabilities_for_considering = (np.array(considering, dtype=float).reshape((1, -1)) @ self.transition_matrix).ravel()
        probabilities_for_considering[~considering] = 0

        average_cardinality = (np.sum(mother) + np.sum(father)) / 2 - np.sum(guaranteed)
        actual_probabilities = scale_to_have_sum(probabilities_for_considering, wanted_sum=average_cardinality)
        actual_probabilities[~considering] = 0
        actual_probabilities[guaranteed] = 1

        child_1 = sample_PS_from_probabilties_for_global(actual_probabilities)
        # child 1 and child 2 are complementary w.r.t. the considering array
        child_2 = np.logical_xor(considering, child_1)

        return (child_1, child_2)


    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape

        children = np.array([self.ps_uniform_crossover(mother, father)
                             for mother, father in zip(X[0], X[1])])

        return np.swapaxes(children, 0, 1)
    def _do_experimental(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape

        def make_child_pair():
            mother = random.choice(X[0])
            father = random.choice(X[1])
            return self.ps_uniform_crossover(mother, father)

        children = np.array([make_child_pair() for _ in range(n_matings)])

        return np.swapaxes(children, 0, 1)



