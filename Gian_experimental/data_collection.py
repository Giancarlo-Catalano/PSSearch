import itertools
import json
import os
from typing import Callable

import numpy as np

import utils
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import MannWhitneyU
from Core.PSMetric.Linkage.Atomicity import Atomicity
from Core.PSMetric.Linkage.ValueSpecificMutualInformation import FasterSolutionSpecificMutualInformation
from Core.get_pRef import get_pRef_from_metaheuristic
from PolishSystem.GlobalPSPolishSearch import find_ps_in_polish_problem
from PolishSystem.OperatorsBasedOnSimilarities.get_operators import get_operators_for_similarities
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import gian_get_similarities
from PolishSystem.read_data import get_pRef_from_vectors
from SimplifiedSystem.ps_search_utils import get_metric_function


def data_collection(train_pRef: PRef,
                    test_pRef: PRef,
                    method: Callable[[PRef], list[PS]]) -> dict:

    # on test
    sample_size_test = get_metric_function("sample_count", pRef=test_pRef)
    fitness_consistency_evaluator_test = MannWhitneyU("greater", sample_threshold=30)
    fitness_consistency_evaluator_test.set_pRef(test_pRef)
    consistency_test = fitness_consistency_evaluator_test.get_single_score

    # on train
    sample_size_train = get_metric_function("sample_count", pRef=train_pRef)
    fitness_consistency_evaluator_train = MannWhitneyU("greater", sample_threshold=30)
    fitness_consistency_evaluator_train.set_pRef(train_pRef)
    consistency_train = fitness_consistency_evaluator_test.get_single_score

    ones_count = lambda x: x.fixed_count()

    pss = method(train_pRef)

    return {"pss": list(map(repr, pss)),
            "sample_sizes_test": list(map(sample_size_test, pss)),
            "consistencies_test": list(map(consistency_test, pss)),
            "sample_sizes_train": list(map(sample_size_train, pss)),
            "consistencies_train": list(map(consistency_train, pss)),
            "ones": list(map(ones_count, pss))
            }


def compare_methods_on_pRef(train_pRef: PRef,
                            test_pRef: PRef,
                            pRef_name: str,
                            methods: dict[str, Callable],
                            file_destination: str):
    data = {"pRef": pRef_name,
            "results": [{"method": method_name,
                         "run_result": data_collection(train_pRef, test_pRef, method)}
                        for method_name, method in methods.items()]}

    with utils.open_and_make_directories(file_destination) as file:
        json.dump(data, file, indent=4)

    print(f"successfully dumped the data into {file_destination}")


def get_data_comparing_operators(seed: int):
    size = 250
    clustering_method = "qmc"
    fitness_column_to_use = 2
    data_folder = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting"

    # pRef = get_pRef_from_vectors(get_vectors_file_name(data_folder, size, clustering_method),
    #                                      get_fitness_file_name(data_folder, size, clustering_method), fitness_column_to_use)

    dir_250 = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting\250"

    def in_250(path):
        return os.path.join(dir_250, path)

    # train_pRef = get_pRef_from_vectors(name_of_vectors_file=in_250("train_many_hot_vectors_250_qmc.csv"),
    #                                       name_of_fitness_file=in_250("train_fitness_250_qmc.csv"),
    #                                   column_in_fitness_file=2)
    # test_pRef = get_pRef_from_vectors(name_of_vectors_file=in_250("test_many_hot_vectors_250_qmc.csv"),
    #                                       name_of_fitness_file=in_250("test_fitness_250_qmc.csv"),
    #                                   column_in_fitness_file=2)

    pRef = get_pRef_from_vectors(name_of_vectors_file=in_250("many_hot_vectors_250_qmc.csv"),
                                 name_of_fitness_file=in_250("fitness_250_qmc.csv"),
                                 column_in_fitness_file=2)

    train_pRef, test_pRef = pRef.train_test_split(test_size=0.3, random_state=seed)

    cluster_info_file_name = in_250(f"cluster_info_{size}_{clustering_method}.pkl")
    similarities = gian_get_similarities(cluster_info_file_name)

    sampler, mutation, crossover = get_operators_for_similarities(similarities, test_pRef,
                                                                  wanted_average_quantity_of_ones=2)

    def atomicity_on_similarity(ps):
        if ps.fixed_count() < 2:
            return 1000
        else:
            valid_indices = [index for index, value in enumerate(ps.values) if value != STAR]
            linkages = [similarities[a, b] for a, b in itertools.combinations(valid_indices, r=2)]
            return -np.average(linkages)

    def make_cached_metric(original_metric):
        cached_values = dict()

        def get_value(ps):
            if ps in cached_values:
                return cached_values[ps]
            else:
                value = original_metric(ps)
                cached_values[ps] = value
                return value

        return get_value

    simplicity = make_cached_metric(get_metric_function("simplicity"))
    sample_size = make_cached_metric(get_metric_function("sample_count", pRef=train_pRef))
    consistency = make_cached_metric(get_metric_function("consistency/greater", pRef=train_pRef))
    atomicity_new = make_cached_metric(atomicity_on_similarity)

    slow_atomicity_metric = Atomicity()
    slow_atomicity_metric.set_pRef(train_pRef)
    slow_atomicity = make_cached_metric(slow_atomicity_metric.get_single_score)

    def control(pRef):
        return find_ps_in_polish_problem(original_problem_search_space=pRef.search_space,
                                         objectives=[sample_size, consistency, slow_atomicity],
                                         ps_budget=10000,
                                         population_size=100,
                                         culling_method=None,
                                         verbose=True,
                                         )

    def with_new_atomicity(pRef):
        return find_ps_in_polish_problem(original_problem_search_space=pRef.search_space,
                                         objectives=[sample_size, consistency, atomicity_new],
                                         ps_budget=10000,
                                         population_size=100,
                                         culling_method=None,
                                         verbose=True,
                                         )

    def with_new_operators(pRef):
        return find_ps_in_polish_problem(original_problem_search_space=pRef.search_space,
                                         objectives=[sample_size, consistency, slow_atomicity],
                                         ps_budget=10000,
                                         population_size=100,
                                         sampling_operator=sampler,
                                         mutation_operator=mutation,
                                         crossover_operator=crossover,
                                         culling_method=None,
                                         verbose=True,
                                         )

    def with_new_atomicity_and_new_operators(pRef):
        return find_ps_in_polish_problem(original_problem_search_space=pRef.search_space,
                                         objectives=[sample_size, consistency, atomicity_new],
                                         ps_budget=10000,
                                         population_size=100,
                                         sampling_operator=sampler,
                                         mutation_operator=mutation,
                                         crossover_operator=crossover,
                                         culling_method=None,
                                         verbose=True,
                                         )

    results_dir = r"C:\Users\gac8\PycharmProjects\PSSearch\Gian_experimental\operator_data_collection\V2"
    compare_methods_on_pRef(train_pRef=train_pRef,
                            test_pRef=test_pRef,
                            pRef_name=f"PRef,cluster_size = {size}, method = {clustering_method}",
                            methods={"control": control,
                                     "with_new_operators": with_new_operators,
                                     "with_new_atomicity_and_new_operators": with_new_atomicity_and_new_operators,
                                     "with_new_atomicity": with_new_atomicity,
                                     },
                            file_destination=os.path.join(results_dir,
                                                          "compare_methods" + utils.get_formatted_timestamp() + ".json"))


def get_data_comparing_operators_dummy():
    problem = RoyalRoad(5)

    # then we make a pRef
    pRef = get_pRef_from_metaheuristic(problem=problem,
                                       sample_size=10000,
                                       which_algorithm="GA",
                                       unique=True,
                                       verbose=True)

    train_pRef, test_pRef = pRef.train_test_split(test_size=0.2)

    evaluator, atomicity_metric = get_metric_function("estimated_atomicity&evaluator", pRef=train_pRef,
                                                      solution=train_pRef.get_best_solution())
    similarities = evaluator.linkage_table

    true_atomicity = get_metric_function("ground_truth_atomicity", problem=problem,
                                         solution=train_pRef.get_best_solution())

    sampler, mutation, crossover = get_operators_for_similarities(similarities, test_pRef,
                                                                  wanted_average_quantity_of_ones=2)

    def atomicity_on_similarity(ps):
        if ps.fixed_count() < 2:
            return 1000
        else:
            valid_indices = [index for index, value in enumerate(ps.values) if value != STAR]
            linkages = [similarities[a, b] for a, b in itertools.combinations(valid_indices, r=2)]
            return -np.average(linkages)

    def make_cached_metric(original_metric):
        cached_values = dict()

        def get_value(ps):
            if ps in cached_values:
                return cached_values[ps]
            else:
                value = original_metric(ps)
                cached_values[ps] = value
                return value

        return get_value

    simplicity = make_cached_metric(get_metric_function("simplicity"))
    sample_size = make_cached_metric(get_metric_function("sample_count", pRef=train_pRef))
    consistency = make_cached_metric(get_metric_function("consistency/greater", pRef=train_pRef))
    mean_fitness = make_cached_metric(get_metric_function("mean_fitness", pRef=pRef))
    estimated_atomicity = make_cached_metric(atomicity_on_similarity)
    # atomicity_old = make_cached_metric(get_metric_function("estimated_atomicity", pRef= train_pRef))
    variance = get_metric_function("variance", pRef=pRef)

    def with_true_atomicity(pRef):
        return find_ps_in_polish_problem(original_problem_search_space=pRef.search_space,
                                         objectives=[simplicity, mean_fitness, true_atomicity],
                                         ps_budget=10000,
                                         population_size=100,
                                         culling_method=None,
                                         verbose=True,
                                         )

    def with_old_atomicity(pRef):
        return find_ps_in_polish_problem(original_problem_search_space=pRef.search_space,
                                         objectives=[simplicity, mean_fitness, estimated_atomicity],
                                         ps_budget=10000,
                                         population_size=100,
                                         culling_method=None,
                                         verbose=True,
                                         )

    # def using_old_operators_old_atomicity(pRef):
    #     return find_ps_in_polish_problem(original_problem_search_space=pRef.search_space,
    #                                     objectives=[sample_size, consistency, atomicity_old],
    #                                     ps_budget=10000,
    #                                     population_size=100,
    #                                     culling_method=None,
    #                                     verbose=True,
    #                                     )

    results_dir = r"C:\Users\gac8\PycharmProjects\PSSearch\Gian_experimental"
    compare_methods_on_pRef(train_pRef=train_pRef,
                            test_pRef=test_pRef,
                            pRef_name=f"RR",
                            methods={"true": with_true_atomicity,
                                     "estimated": with_old_atomicity,
                                     },
                            file_destination=os.path.join(results_dir,
                                                          "compare_methods" + utils.get_formatted_timestamp() + ".json"))



trials = 20
for trial in range(trials):
    get_data_comparing_operators(trial)
