import numpy as np
from pymoo.operators.sampling.rnd import FloatRandomSampling

from Core.PRef import PRef
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import \
    sample_PS_from_probabilties_for_global, scale_to_have_sum


class DistributionSampling(FloatRandomSampling):

    n: int
    probabilities: np.ndarray

    def __init__(self, pRef: PRef, wanted_average_quantity_of_ones: float):
        super().__init__()
        self.n = pRef.search_space.amount_of_parameters
        distribution = np.average(pRef.full_solution_matrix, axis=0)
        self.probabilities = scale_to_have_sum(distribution, wanted_sum=wanted_average_quantity_of_ones)

    def generate_single_individual(self, n) -> np.ndarray:
        return sample_PS_from_probabilties_for_global(self.probabilities)

    def _do(self, problem, n_samples, **kwargs):
        n = problem.n_var
        return np.array([self.generate_single_individual(n) for _ in range(n_samples)])