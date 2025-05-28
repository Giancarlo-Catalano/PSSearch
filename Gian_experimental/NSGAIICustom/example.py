import heapq
import itertools
import math
import os
from typing import Iterable, Callable

import numpy as np
from scipy.stats import mannwhitneyu

import utils
from Core.PRef import PRef
from Core.PS import PS
from Gian_experimental.NSGAIICustom.NSGAIICustom import NSGAIICustom, NCSolution, NCSamplerSimple, NCMutationSimple, \
    NCCrossoverSimple, EvaluatedNCSolution, NCSamplerFromPRef, NCCrossoverTransition, NCMutation, \
    NCMutationCounterproductive
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import get_transition_matrix
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import gian_get_similarities
from PolishSystem.read_data import get_pRef_from_vectors


def complexity(ps):
    return len(ps)


def make_similarity_atomicity(similarities):
    def atomicity(ps):
        if len(ps) < 2:
            return -1000
        else:
            linkages = [similarities[a, b] for a, b in itertools.combinations(ps, r=2)]
            return np.average(linkages)

    return atomicity


def get_matches_non_matches_in_pRef(ps, given_pRef: PRef):
    ps_true_values = np.full(shape=250, fill_value=-1, dtype=int)
    ps_true_values[list(ps)] = 1
    return given_pRef.fitnesses_of_observations_and_complement(PS(ps_true_values))


def make_consistency_metric_with_sample_size(pRef: PRef,
                                             threshold: float = 0.5,
                                             must_match_at_least: int = 3):
    def consistency_and_sample(ps):
        # matches, non_matches = sPRef.get_matching_fitnesses_and_not_matching(ps, threshold=threshold)
        matches, non_matches = get_matches_non_matches_in_pRef(ps, pRef)
        if min(len(matches), len(non_matches)) < must_match_at_least:
            return 1, len(matches)
        else:
            test = mannwhitneyu(matches, non_matches, alternative="greater", method="asymptotic")
            return test.pvalue, len(matches)
            # return permutation_mannwhitney_u(matches, non_matches, n_permutations=50), len(matches)

    return consistency_and_sample


def make_min_metric_with_sample_size(pRef: PRef):
    def min_and_sample(ps):
        matches, non_matches = get_matches_non_matches_in_pRef(ps, pRef)
        if min(len(matches), len(non_matches)) < 1:
            return (-1000, len(matches))
        else:
            lowest_fitness = np.min(matches)
        return lowest_fitness, len(matches)

    return min_and_sample


class HashedSolution:
    solution: NCSolution

    def __init__(self,
                 sol):
        self.solution = sol

    def __hash__(self):
        return sum(self.solution) % 7787

    def __eq__(self, other):
        return self.solution == other.solution


def make_metrics_cached(metrics):
    cached_values = dict()

    def get_values(ps):
        wrapped = HashedSolution(ps)
        if wrapped in cached_values:
            return cached_values[wrapped]
        else:

            value = metrics(ps)
            cached_values[wrapped] = value
            return value

    return get_values


dir_250 = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting\250"


def in_250(path):
    return os.path.join(dir_250, path)


def count_frequencies(iterable):
    iterable_list = list(iterable)
    counts = {item: iterable_list.count(item)
              for item in set(iterable)}

    for key, count in counts.items():
        print(key, count)


def main():
    with utils.announce("Loading the pRef and other data"):
        pRef = get_pRef_from_vectors(name_of_vectors_file=in_250("many_hot_vectors_250_random.csv"),
                                     name_of_fitness_file=in_250("fitness_250_random.csv"),
                                     column_in_fitness_file=2)

        train_pRef, test_pRef = pRef.train_test_split(test_size=0.3)

        print(f"{train_pRef.sample_size = }, {test_pRef.sample_size = }")
        cluster_info_file_name = in_250(f"cluster_info_250_qmc.pkl")
        similarities = gian_get_similarities(cluster_info_file_name)
        n = pRef.full_solution_matrix.shape[1]



    with utils.announce("Printing the stats for each variable"):
        print(f"The pRef has {pRef.sample_size}, where the variables appear")
        quantity_by_variable = [(index, np.sum(pRef.full_solution_matrix[:, index]))
                                for index in range(n)]
        top_10_most_common = heapq.nlargest(n = 10, iterable=quantity_by_variable, key=utils.second)
        print("The top 10 most common vars are: "+", ".join(f"{var =}:{samples = }" for var, samples in top_10_most_common))

    with utils.announce("Making the operators and metrics"):
        custom_sampling = NCSamplerFromPRef.from_PRef(train_pRef)
        transition_matrix = get_transition_matrix(similarities)
        custom_crossover = NCCrossoverTransition(transition_matrix)
        custom_mutation = NCMutationCounterproductive(transition_matrix, disappearance_probability=0.9)

        # train_mean_fitness = make_mean_fitness(train_pRef, threshold=threshold)
        train_atomicity = make_similarity_atomicity(similarities)
        train_consistency_and_sample = make_consistency_metric_with_sample_size(train_pRef, must_match_at_least=3)
        train_min_and_sample: Callable[[NCSolution], (float, float)] = make_min_metric_with_sample_size(train_pRef)
        test_consistency_and_sample = make_consistency_metric_with_sample_size(test_pRef, must_match_at_least=3)



        traditional_sampling = NCSamplerSimple.with_average_quantity(3, genome_size=n)
        traditional_mutation = NCMutationSimple(n)

        traditional_crossover = NCCrossoverSimple(swap_probability=1 / n)

    with utils.announce("Constructing the algorithm"):
        def get_metrics(ps: NCSolution) -> tuple[float]:
            p_value, sample_size = train_consistency_and_sample(ps)
            atomicity = train_atomicity(ps)
            return (-sample_size, p_value, -atomicity)

        def keep_ones_with_most_samples(population: Iterable[EvaluatedNCSolution], quantity_required: int):
            return heapq.nsmallest(iterable=population, key=lambda x: x.fitnesses[0], n=quantity_required)

        algorithm = NSGAIICustom(sampling=custom_sampling,
                                 mutation=custom_mutation,
                                 crossover=custom_crossover,
                                 probability_of_crossover=0.5,
                                 eval_budget=10000,
                                 pop_size=100,
                                 tournament_size=3,
                                 mo_fitness_function=make_metrics_cached(get_metrics),
                                 unique=True,
                                 verbose=True,
                                 culler=keep_ones_with_most_samples)

    with utils.announce("Running the algorithm"):
        results = algorithm.run()

    with utils.announce("Gathering data"):

        p_value_combinations = {(on_train, on_test): 0
                                for on_train in [True, False]

                                for on_test in [True, False]}
        for ps in results:
            train_p, train_sample = train_consistency_and_sample(ps.solution)
            test_p, test_sample = test_consistency_and_sample(ps.solution)
            print(ps, f"{train_p = }, {test_p = }, {train_sample = }, {test_sample = }")
            p_value_combinations[(train_p < 0.05, test_p < 0.05)] += 1


        for key in p_value_combinations:
            print(key, p_value_combinations[key])


main()
