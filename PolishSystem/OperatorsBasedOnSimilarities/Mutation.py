import numpy as np
from pymoo.core.mutation import Mutation

from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import \
    sample_PS_from_probabilties_for_global, scale_to_have_sum_and_max


class TransitionMutation(Mutation):
    n: int
    single_point_probability: float
    disappearance_rate: float
    transition_matrix: np.ndarray

    def __init__(self,
                 single_point_probability: float,
                 transition_matrix: np.ndarray, prob=None):
        self.n = transition_matrix.shape[0]
        self.transition_matrix = transition_matrix
        self.single_point_probability = single_point_probability
        self.disappearance_rate = 1-single_point_probability
        super().__init__(
            prob=0.9 if prob is None else prob)  # no idea what's supposed to be there, but it used to say 0.9 by default..

    def mutate_single_individual(self, x: np.ndarray) -> np.ndarray:
        values = x.copy()
        # step 1: obtain probabilties from the transition matrix (as if it was a markov model)
        unscaled_probabilities = (values.reshape((1, -1)) @ self.transition_matrix).reshape(-1)

        # step 2: scale the probabilities so that
        ## * the expected quantity of ones at the end is probably the same as the original (wanted sum)
        ## * the probability of the original values staying is wanted_max

        probabilities = scale_to_have_sum_and_max(unscaled_probabilities, wanted_sum=np.sum(values),
                                                  wanted_max=self.disappearance_rate, positions=self.n)
        # step 3: generate a solution based on those probabilities
        return sample_PS_from_probabilties_for_global(probabilities)

    def _do(self, problem, X, params=None, **kwargs):
        result_values = X.copy()
        for index, row in enumerate(result_values):
            result_values[index] = self.mutate_single_individual(row)

        return result_values
