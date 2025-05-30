from BenchmarkProblems.BT.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.FullSolution import FullSolution
from typing import Callable
from Core.SearchSpace import SearchSpace
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem

import json
from DetectingAnnoyingEffects.detect import ProblemSearchSpace, BooleanVariable, detect_effects_on_dynamic_dataset, \
    DynamicFitnessFunctionWithRandomSample, AnnoyingEffects, CombinatorialVariable
import numpy as np
import utils
from Core.FullSolution import FullSolution

class SubProblem:
    search_space: SearchSpace

    def __init__(self,
                 search_space: SearchSpace):
        self.search_space = search_space

    def fitness_function(self, solution):
        raise NotImplemented()


class Univariate(SubProblem):
    fitnesses: list[float]

    def __init__(self, fitnesses: list[float]):
        super().__init__(SearchSpace([len(fitnesses)]))
        self.fitnesses = fitnesses

    def fitness_function(self, solution):
        return self.fitnesses[solution[0]]


class UnitaryProblem(SubProblem):
    clique_size: int

    def __init__(self,
                 clique_size: int):
        super().__init__(SearchSpace([2] * clique_size))
        self.clique_size = clique_size

    def fitness_function(self, solution):
        ones = sum(solution)
        return self.unitary_function(ones)

    def unitary_function(self, n):
        raise NotImplemented


class OneMax(UnitaryProblem):
    def __init__(self, clique_size: int):
        super().__init__(clique_size)

    def unitary_function(self, n):
        return n


class RoyalRoad(UnitaryProblem):

    def __init__(self, clique_size: int):
        super().__init__(clique_size)

    def unitary_function(self, n):
        return self.clique_size if n == self.clique_size else 0


class Parity(UnitaryProblem):
    def __init__(self, clique_size: int):
        super().__init__(clique_size)

    def unitary_function(self, n):
        return n % 2


class TwoPeaks(UnitaryProblem):
    def __init__(self, clique_size: int):
        super().__init__(clique_size)

    def unitary_function(self, n):
        return max(n, self.clique_size - n)


class Trapk(UnitaryProblem):
    def __init__(self, clique_size: int):
        super().__init__(clique_size)

    def unitary_function(self, n):
        return self.clique_size if n == self.clique_size else self.clique_size - 1 - n


class NITProblem(BenchmarkProblem):
    sub_problems: list[SubProblem]
    aggregation_func: Callable

    def __init__(self, sub_problems, aggregation_func):
        self.sub_problems = sub_problems
        self.aggregation_func = aggregation_func
        super().__init__(SearchSpace.concatenate_search_spaces(sub.search_space for sub in self.sub_problems))

    def fitness_function(self, fs: FullSolution) -> float:
        next_start = 0
        fragments = []
        for problem in self.sub_problems:
            end = next_start + problem.search_space.amount_of_parameters
            fragments.append(fs.values[next_start:end])
            next_start = end

        sub_fitnesses = [problem.fitness_function(fragment) for problem, fragment in zip(self.sub_problems, fragments)]
        return self.aggregation_func(sub_fitnesses)



def check_on_artificial_problems():


    problem = NITProblem(
        [Univariate([2, 3, -40]), Univariate([0, 10]), RoyalRoad(4), Trapk(5), OneMax(4), Parity(3), TwoPeaks(4)],
        aggregation_func=sum)

    search_space = ProblemSearchSpace([CombinatorialVariable(card) for card in problem.search_space.cardinalities])

    def fitness_function(sol):
        # wraps it into a Solution
        return problem.fitness_function(FullSolution(sol))



    report = detect_effects_on_dynamic_dataset(
        dynamic_fitness_function=DynamicFitnessFunctionWithRandomSample(fitness_function=fitness_function,
                                                                        search_space=search_space,
                                                                        quantity_of_samples=100,
                                                                        to_maximise=True),
        which_variables=list(range(problem.search_space.amount_of_parameters)),
        which_effects=[AnnoyingEffects.NON_MARGINALITY]
    )

    print(json.dumps(report, indent=4))


def check_on_BT_problem():
    problem = EfficientBTProblem.from_default_files()
    search_space = ProblemSearchSpace([CombinatorialVariable(card) for card in problem.search_space.cardinalities])

    def fitness_function(sol):
        # wraps it into a Solution
        return problem.fitness_function(FullSolution(sol))

    report = detect_effects_on_dynamic_dataset(
        dynamic_fitness_function=DynamicFitnessFunctionWithRandomSample(fitness_function=fitness_function,
                                                                        search_space=search_space,
                                                                        quantity_of_samples=100,
                                                                        to_maximise=True),
        which_variables=list(range(problem.search_space.amount_of_parameters)),
        which_effects=[AnnoyingEffects.NON_MARGINALITY]
    )

    print(json.dumps(report, indent=4))



check_on_BT_problem()
