import numpy as np
from pymoo.core.crossover import Crossover

from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import scale_to_have_sum_and_max, \
    sample_PS_from_probabilties_for_global


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

    @classmethod
    def ps_uniform_crossover(self, mother: np.ndarray, father: np.ndarray):
        guaranteed = mother * father
        considering = mother + father - guaranteed * 2

        guaranteed = np.array(guaranteed, dtype=bool)
        considering = np.array(considering, dtype=bool)

        probabilities_for_considering = (considering.reshape((1, -1)) @ self.transition_matrix).ravel()
        probabilities_for_considering[~considering] = 0

        average_cardinality = (np.sum(mother) + np.sum(father)) / 2 - np.sum(guaranteed)
        actual_probabilities = scale_to_have_sum_and_max(probabilities_for_considering, wanted_sum=average_cardinality,
                                                         wanted_max=0.5, positions=len(considering))
        actual_probabilities[~considering] = 0
        actual_probabilities[guaranteed] = 1

        return sample_PS_from_probabilties_for_global(actual_probabilities)

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape

        children = np.array([self.ps_uniform_crossover(mother, father)
                             for mother, father in zip(X[0], X[1])])

        return np.swapaxes(children, 0, 1)



