import itertools
import random
from enum import Enum
from typing import Literal, Optional, Callable, Iterable

import numpy as np


## this is a stub

class SearchSpaceVariable:

    def __init__(self):
        pass

    def __repr__(self):
        raise NotImplementedError()

    def random(self):
        raise NotImplementedError()


class CombinatorialVariable(SearchSpaceVariable):
    cardinality: int
    as_strings: tuple[str, ...]

    def __init__(self, cardinality: int,
                 as_strings: Optional[Iterable[str, ...]] = None):
        super().__init__()
        assert (cardinality > 0)
        assert (isinstance(cardinality, int))

        self.cardinality = cardinality
        if as_strings is None:
            self.as_strings = tuple(map(str, range(cardinality)))
        else:
            self.as_strings = tuple(as_strings)
            assert (len(self.as_strings) == self.cardinality)

    def repr(self, value: int) -> str:
        return self.as_strings[value]

    def random(self) -> int:
        return random.randrange(self.cardinality)

    def random_different(self, original_value: int) -> int:
        value = random.randrange(self.cardinality - 1)
        return value + (value >= original_value)


class BooleanVariable(CombinatorialVariable):
    def __init__(self):
        super().__init__(cardinality=2)


class ContinuousVariable:
    minimum: float
    maximum: float

    str_format: str

    def __init__(self,
                 minimum: float,
                 maximum: float,
                 str_format: Optional[str]):
        assert (minimum < maximum)
        self.minimum = minimum
        self.maximum = maximum
        self.str_format = ".>4.2f" if str_format is None else str_format

    def repr(self, value: float) -> str:
        return f"{value:{self.str_format}}"

    def random(self) -> float:
        return random.uniform(self.minimum, self.maximum)


Solution = np.ndarray
FitnessFunction = Callable[[Solution], float]


class ProblemSearchSpace:
    variables: tuple[SearchSpaceVariable, ...]

    def __init__(self, variables: Iterable[SearchSpaceVariable]):
        self.variables = tuple(variables)

    def repr(self, solution: Solution) -> str:
        assert (len(solution) == len(self.variables))
        return " ".join(variable.repr(value) for variable, value in zip(self.variables, solution))

    def random(self) -> Solution:
        return np.fromiter((variable.random() for variable in self.variables), dtype=float)

    def enumerate(self) -> list[Solution]:
        assert (all(isinstance(variable, CombinatorialVariable) for variable in self.variables))
        ranges = [range(variable.cardinality) for variable in self.variables]
        return list(map(np.array, itertools.product(*ranges)))


class FitnessFunctionUsage:
    pass

    def get_sample_of_solutions(self) -> (list[Solution], list[float]):
        raise NotImplementedError()


class DynamicFitnessFunction(FitnessFunctionUsage):
    fitness_function: FitnessFunction
    search_space: ProblemSearchSpace

    def __init__(self,
                 fitness_function: FitnessFunction,
                 search_space: ProblemSearchSpace):
        super().__init__()
        self.fitness_function = fitness_function
        self.search_space = search_space

    def with_fitness_values(self, solutions: list[Solution]) -> (list[Solution], list[float]):
        fitness_values = list(map(self.fitness_function, solutions))
        return solutions, fitness_values

    def get_sample_of_solutions(self) -> (list[Solution], list[float]):
        raise NotImplementedError()


class DynamicFitnessFunctionWithEnumeration(DynamicFitnessFunction):

    def __init__(self,
                 fitness_function: FitnessFunction,
                 search_space: ProblemSearchSpace):
        super().__init__(fitness_function, search_space)

    def get_sample_of_solutions(self) -> (list[Solution], list[float]):
        return self.with_fitness_values(self.search_space.enumerate())


class DynamicFitnessFunctionWithRandomSample(DynamicFitnessFunction):
    quantity_of_samples: int

    def __init__(self,
                 fitness_function: FitnessFunction,
                 search_space: ProblemSearchSpace,
                 quantity_of_samples: int):
        super().__init__(fitness_function, search_space)
        self.quantity_of_samples = quantity_of_samples

    def get_sample_of_solutions(self) -> (list[Solution], list[float]):
        solutions = [self.search_space.random() for _ in range(self.quantity_of_samples)]
        return self.with_fitness_values(solutions)


class StaticFitnessFunction(FitnessFunctionUsage):
    solutions: list[Solution]
    fitness_values: np.ndarray

    def __init__(self, solutions: list[Solution],
                 fitness_values: Iterable[float]):
        super().__init__()

        self.solutions = solutions
        self.fitness_values = np.array(fitness_values)
        assert (len(self.solutions) == len(self.fitness_values))

    def get_sample_of_solutions(self) -> (list[Solution], list[float]):
        return self.solutions, self.fitness_values


class AnnoyingEffects(Enum):
    NON_MARGINALITY = 1
    MORE_THAN_BIVARIATE = 2
    SOLUTION_DEPENDENT_INTERACTIONS = 3
    DECEPTIVE_BEHAVIOUR = 4
    HITCHHIKING = 5


all_annoying_effects = [AnnoyingEffects.NON_MARGINALITY, AnnoyingEffects.MORE_THAN_BIVARIATE,
                        AnnoyingEffects.SOLUTION_DEPENDENT_INTERACTIONS,
                        AnnoyingEffects.DECEPTIVE_BEHAVIOUR,
                        AnnoyingEffects.HITCHHIKING]


def detect_non_marginality_on_dynamic_dataset():
    pass


def detect_effects_on_static_dataset(which_effects: list[AnnoyingEffects],
                                     which_variables: list[int],
                                     static_fitness_function: StaticFitnessFunction):
    pass


def detect_effects_on_dynamic_dataset(which_effects: list[AnnoyingEffects],
                                      which_variables: list[int],
                                      dynamic_fitness_function: DynamicFitnessFunction):
    pass


def detect_effects(which: str = "all",
                   fitness_function_availability: Literal["dynamic", "static"] = "dynamic",
                   enumerate_fully: Optional[bool] = False,
                   fitness_function: Callable[[np.ndarray], float] = None):
    pass
