import itertools
from math import ceil
from typing import Optional, Literal

import numpy as np
from pandas.io.common import file_exists
from scipy.stats import t

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from FSStochasticSearch.HistoryPRefs import uniformly_random_distribution_pRef, pRef_from_GA, pRef_from_SA, \
    pRef_from_tabu_search
from utils import announce


def get_pRef_from_metaheuristic_unsafe(benchmark_problem: BenchmarkProblem,
                                       sample_size: int,
                                       which_algorithm: Literal["uniform", "GA", "SA", "tabu"],
                                       verbose=True):
    with announce(f"Running the algorithm to generate the PRef using {which_algorithm}", verbose=verbose):
        match which_algorithm:
            case "uniform":
                return uniformly_random_distribution_pRef(sample_size=sample_size,
                                                          benchmark_problem=benchmark_problem)
            case "GA":
                return pRef_from_GA(benchmark_problem=benchmark_problem,
                                    sample_size=sample_size,
                                    ga_population_size=100)
            case "SA":
                return pRef_from_SA(benchmark_problem=benchmark_problem,
                                    sample_size=sample_size,
                                    max_trace=sample_size)
            case "Tabu":
                return pRef_from_tabu_search(benchmark_problem=benchmark_problem,
                                             sample_size=sample_size,
                                             max_trace=sample_size)
            case _:
                raise ValueError


def get_pRef_from_metaheuristic(problem,
                                sample_size: int,
                                which_algorithm: str,
                                unique: bool = True,
                                force_include: Optional[list[FullSolution]] = None,
                                verbose: bool = False) -> PRef:
    methods = which_algorithm.split()
    sample_size_for_each = ceil(sample_size / len(methods))

    def make_pRef_with_method(method: str) -> PRef:
        return get_pRef_from_metaheuristic_unsafe(benchmark_problem=problem,
                                                  which_algorithm=method,
                                                  sample_size=sample_size_for_each,
                                                  verbose=verbose)

    pRefs = [make_pRef_with_method(method) for method in methods]

    if force_include is not None and len(force_include) > 0:
        forced_pRef = PRef.from_full_solutions(force_include,
                                               fitness_values=[problem.fitness_function(fs) for fs in
                                                               force_include],
                                               search_space=problem.search_space)
        pRefs.append(forced_pRef)

    final_pRef = PRef.concat(pRefs)
    if unique:
        final_pRef = PRef.unique(final_pRef)
    return final_pRef
