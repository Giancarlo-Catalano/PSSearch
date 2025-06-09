import itertools

import numpy as np

from DetectingAnnoyingEffects.MarginalityDetection.CombinatorialProblems.CombinatorialProblem import \
    CombinatorialProblem


def detect_marginality_of_combinatorial(problem: CombinatorialProblem,
                                        samples_to_test_on: list[np.ndarray]):
    def get_domination_from_fitnesses(fitnesses):
        return [np.sum(fitnesses > fitness) for fitness in fitnesses]

    def get_domination_for_solution(solution: np.ndarray, var: int):
        cp = []
        for perturbed_value in range(problem.cardinalities[var]):
            perturbed_solution = solution.copy()
            perturbed_solution[var] = perturbed_value
            fitness = problem.fitness_function(perturbed_solution)
            cp.append(fitness)

        return get_domination_from_fitnesses(cp)

    def distance_metric_between_dominations(dom_a, dom_b):
        return np.sum(np.abs(dom_a - dom_b)) / 2

    def get_average_domination_diff_for_var(var):
        print(f"Analysing var {var}")
        dominations = np.array([get_domination_for_solution(solution, var)
                                for solution in samples_to_test_on])

        return np.average([distance_metric_between_dominations(a, b)
                           for a, b in itertools.combinations(dominations, r=2)])

    return np.array([get_average_domination_diff_for_var(var)
                     for var in range(problem.n)])
