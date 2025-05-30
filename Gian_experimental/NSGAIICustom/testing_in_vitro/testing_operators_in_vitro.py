import heapq
import itertools
import json
import random
from typing import Iterable, Optional, Iterator

import numpy as np

import utils
from Gian_experimental.NSGAIICustom.NSGAIICustom import NSGAIICustom, NCSolution, EvaluatedNCSolution, NCSampler, \
    NCMutation, NCCrossover, NCSamplerFromPRef, NCCrossoverTransition, NCSamplerSimple, NCMutationSimple, \
    NCCrossoverSimple, NCMutationCounterproductive
from Gian_experimental.NSGAIICustom.testing_in_vitro.SPRef import SPRef, OptimisedSPref
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import gian_get_similarities, get_transition_matrix
from PolishSystem.read_data import get_pRef_from_vectors
import os
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import get_transition_matrix
from typing import Iterable
from Gian_experimental.NSGAIICustom.NSGAIICustom import NSGAIICustom, NCSolution, NCSamplerSimple, NCMutationSimple, \
    NCCrossoverSimple, EvaluatedNCSolution, NCSamplerFromPRef, NCCrossoverTransition
import heapq


# we want to show how the operators affect the sample_size of the children as the program progresses

def run_NSGAII_in_steps(algorithm: NSGAIICustom) -> Iterable[Iterable[EvaluatedNCSolution]]:
    # this will yield every generation

    used_evaluations = [0]

    def with_fitnesses(solution: NCSolution) -> EvaluatedNCSolution:
        fitnesses = algorithm.mo_fitness_function(solution)
        used_evaluations[0] += 1
        return EvaluatedNCSolution(solution, fitnesses)

    def sampler_yielder():
        while True:
            yield with_fitnesses(algorithm.sampling.sample())

    algorithm.log("Beginning of NC process")
    population = algorithm.make_population(yielder=sampler_yielder(), required_quantity=algorithm.pop_size)
    yield population

    while (used_evaluations[0] < algorithm.eval_budget):
        population = algorithm.make_next_generation(population, with_fitnesses)
        yield population
        algorithm.log(f"Used evals: {used_evaluations[0]}")
    pareto_fronts = algorithm.get_pareto_fronts(population)

    if algorithm.unique:
        return list(set(pareto_fronts[0]))
    yield pareto_fronts[0]


def make_generation_magic_hat(population: Iterable[EvaluatedNCSolution],
                              algorithm: NSGAIICustom,
                              ) -> Iterator[NCSolution]:
    pareto_fronts = algorithm.get_pareto_fronts(population)
    indices_and_ranks = [(index, rank)
                         for rank, front in enumerate(pareto_fronts)
                         for index, _ in enumerate(front)]

    def tournament_select_one() -> EvaluatedNCSolution:
        candidates = random.choices(indices_and_ranks, k=algorithm.tournament_size)
        winner_index, winner_pareto_index = min(candidates, key=utils.second)
        return pareto_fronts[winner_pareto_index][winner_index]

    while True:
        yield tournament_select_one().solution


def repeat(generator, n):
    return [generator() for _ in range(n)]


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


def gather_data(genome_threshold: Optional[float]):
    dir_250 = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting\250"
    # dir_250 = r"/Users/gian/PycharmProjects/PSSearch/data/retail_forecasting/250"

    def in_250(path):
        return os.path.join(dir_250, path)

    train_pRef = get_pRef_from_vectors(name_of_vectors_file=in_250("train_many_hot_vectors_250_qmc.csv"),
                                       name_of_fitness_file=in_250("train_fitness_250_qmc.csv"),
                                       column_in_fitness_file=2)
    test_pRef = get_pRef_from_vectors(name_of_vectors_file=in_250("test_many_hot_vectors_250_qmc.csv"),
                                      name_of_fitness_file=in_250("test_fitness_250_qmc.csv"),
                                      column_in_fitness_file=2)

    train_SPRef = OptimisedSPref.from_pRef(train_pRef)
    test_SPRef = OptimisedSPref.from_pRef(test_pRef)

    cluster_info_file_name = in_250(f"cluster_info_250_qmc.pkl")
    similarities = gian_get_similarities(cluster_info_file_name)

    n = 250

    def keep_ones_with_most_samples(population: Iterable[EvaluatedNCSolution], quantity_required: int):
        return heapq.nsmallest(iterable=population, key=lambda x: x.fitnesses[0], n=quantity_required)

    transition_matrix = get_transition_matrix(similarities)
    custom_sampling = NCSamplerFromPRef.from_PRef(train_pRef)
    custom_mutation = NCMutationCounterproductive(transition_matrix)
    custom_crossover = NCCrossoverTransition(transition_matrix)

    def atomicity(ps):
        if len(ps) < 2:
            return -1000
        else:
            linkages = [similarities[a, b] for a, b in itertools.combinations(ps, r=2)]
            return np.average(linkages)

    traditional_sampling = NCSamplerSimple.with_average_quantity(3, genome_size=n)
    traditional_mutation = NCMutationSimple(n)
    traditional_crossover = NCCrossoverSimple(swap_probability=1 / n)


    def get_metrics(ps: NCSolution) -> tuple[float]:
        # I will use sample size, variance and atomicity

        # for now, simplicity, mean fitness and atomicity
        simplicity_score = len(ps)
        matching, non_matching = train_SPRef.partition(ps, threshold=genome_threshold)
        len_m, len_nm = len(matching), len(non_matching)
        weighted_variance_score = (np.var(matching) * len_m + np.var(non_matching) * len_nm) / (len_m + len_nm)

        atomicity_score = atomicity(ps)

        return (-len(matching), weighted_variance_score, -atomicity_score)

    algorithm = NSGAIICustom(sampling=traditional_sampling,
                             mutation=traditional_mutation,
                             crossover=traditional_crossover,
                             probability_of_crossover=0.5,
                             eval_budget=5000,
                             pop_size=100,
                             tournament_size=3,
                             mo_fitness_function=make_metrics_cached(get_metrics),
                             unique=True,
                             verbose=True,
                             culler=keep_ones_with_most_samples
                             )

    results_by_generation = []

    def sample_count(individual):
        matches, non_matches = test_SPRef.partition(individual, threshold=genome_threshold)
        return len(matches)

    for generation_index, generation in enumerate(run_NSGAII_in_steps(algorithm)):
        magic_hat = make_generation_magic_hat(generation, algorithm)

        children_to_generate = 100
        children_from_tm = repeat(lambda: sample_count(traditional_mutation.mutated(next(magic_hat))),
                                  children_to_generate)
        children_from_cm = repeat(lambda: sample_count(custom_mutation.mutated(next(magic_hat))), children_to_generate)

        children_from_tc, children_from_cc = [], []
        for _ in range(children_to_generate // 2):
            tc_children = traditional_crossover.crossed(next(magic_hat), next(magic_hat))
            cc_children = custom_crossover.crossed(next(magic_hat), next(magic_hat))
            children_from_tc.extend(map(sample_count, tc_children))
            children_from_cc.extend(map(sample_count, cc_children))


        results_by_generation.append({"generation": generation_index,
                                      "genome_threshold": genome_threshold,
                                      "tm": children_from_tm,
                                      "cm": children_from_cm,
                                      "tc": children_from_tc,
                                      "cc": children_from_cc})


    results_path = r"C:\Users\gac8\PycharmProjects\PSSearch\Gian_experimental\NSGAIICustom\testing_in_vitro\v4"
    result_filename = os.path.join(results_path, utils.get_formatted_timestamp()+"in_vitro.json")
    with open(result_filename, "w") as file:
        json.dump(results_by_generation, file, indent=4)

#
# for genome_threshold in [None, 1, 2, 3, 4]:
#     for _ in range(50):
#         gather_data(genome_threshold)
