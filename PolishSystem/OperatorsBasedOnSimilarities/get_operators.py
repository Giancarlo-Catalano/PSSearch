# just a simple helper to get them all at the same time
from Core.PRef import PRef
from PolishSystem.OperatorsBasedOnSimilarities.Crossover import TransitionCrossover
from PolishSystem.OperatorsBasedOnSimilarities.Mutation import TransitionMutation
from PolishSystem.OperatorsBasedOnSimilarities.Sampling import DistributionSampling
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import get_transition_matrix


def get_operators_for_similarities(similarities, pRef: PRef, mutation_rate):
    transition_matrix = get_transition_matrix(similarities)

    sampler = DistributionSampling(pRef)
    mutation = TransitionMutation(single_point_probability=mutation_rate,
                                  transition_matrix=transition_matrix)
    crossover = TransitionCrossover(transition_matrix)
    return sampler, mutation, crossover