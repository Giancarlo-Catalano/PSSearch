from typing import Iterable, Callable

import numpy as np


class CombinatorialProblem:
    cardinalities: np.ndarray
    n: int

    def __init__(self, cardinalities):
        self.n = len(cardinalities)
        self.cardinalities = cardinalities

    def fitness_function(self, x):
        raise NotImplementedError()

    def random_solution(self):
        return np.random.randint(0, self.cardinalities)


class CombinedCombinatorialProblem(CombinatorialProblem):
    original_problems: tuple[CombinatorialProblem, ...]
    aggregation_function: Callable
    starts: np.ndarray

    def __init__(self,
                 original_problems: Iterable[CombinatorialProblem],
                 aggregation_func: Callable):
        self.original_problems = tuple(original_problems)
        cardinalities = np.hstack([problem.cardinalities for problem in self.original_problems])
        self.starts = np.cumsum([0] + [problem.n for problem in self.original_problems])
        super().__init__(cardinalities)
        self.aggregation_function = aggregation_func


    def fitness_function(self, x):
        fitnesses = []
        for start, end, subproblem in zip(self.starts, self.starts[1:], self.original_problems):
            sub_solution = x[start:end]
            fitnesses.append(subproblem.fitness_function(sub_solution))
        return self.aggregation_function(fitnesses)



