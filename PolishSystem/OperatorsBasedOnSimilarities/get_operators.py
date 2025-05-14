# just a simple helper to get them all at the same time
from typing import Optional

import numpy as np

from Core.PRef import PRef
from PolishSystem.OperatorsBasedOnSimilarities.Crossover import TransitionCrossover
from PolishSystem.OperatorsBasedOnSimilarities.Mutation import TransitionMutation
from PolishSystem.OperatorsBasedOnSimilarities.Sampling import DistributionSampling
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import get_transition_matrix


def get_operators_for_similarities(similarities,
                                   pRef: PRef,
                                   wanted_average_quantity_of_ones: Optional[float] = None,
                                   mutation_rate: Optional[float] = None):
    transition_matrix = get_transition_matrix(similarities)

    if wanted_average_quantity_of_ones is None:
        wanted_average_quantity_of_ones = np.average(np.sum(pRef.full_solution_matrix, axis=1))
    sampler = DistributionSampling(pRef, wanted_average_quantity_of_ones)

    if mutation_rate is None:
        mutation_rate = 1 / pRef.search_space.amount_of_parameters
    mutation = TransitionMutation(single_point_probability=mutation_rate,
                                  transition_matrix=transition_matrix)
    crossover = TransitionCrossover(transition_matrix)
    return sampler, mutation, crossover


