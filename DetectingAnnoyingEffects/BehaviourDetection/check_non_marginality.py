from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from BenchmarkProblems.BT.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.NK import NK
from typing import Callable
from Core.SearchSpace import SearchSpace
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem

import json
from DetectingAnnoyingEffects.BehaviourDetection.detect import ProblemSearchSpace, detect_effects_on_dynamic_dataset, \
    DynamicFitnessFunctionWithRandomSample, AnnoyingEffects, CombinatorialVariable
import numpy as np
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


def check_on_graph_colouring():
    problem = GraphColouring.random(amount_of_colours=3, amount_of_nodes=6, chance_of_connection=0.3)
    print(problem.connections)

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


def check_on_NK():
    n = 12
    for k in range(5):
        problem = NK.random(n, k)
        print(f"For problem with k = {k}")

        print(f"The fitness function is the following")
        for var, sub_function in enumerate(problem.fitness_functions):
            print(f"\tSubfunction starting at {var}")
            for value, fitness in enumerate(sub_function):
                print("\t\t", bin(value), fitness)


        search_space = ProblemSearchSpace([CombinatorialVariable(card) for card in problem.search_space.cardinalities])



        def fitness_function(sol):
            # wraps it into a Solution
            return problem.fitness_function(FullSolution(sol))

        report = detect_effects_on_dynamic_dataset(
            dynamic_fitness_function=DynamicFitnessFunctionWithRandomSample(fitness_function=fitness_function,
                                                                            search_space=search_space,
                                                                            quantity_of_samples=1000,
                                                                            to_maximise=True),
            which_variables=list(range(problem.search_space.amount_of_parameters)),
            which_effects=[AnnoyingEffects.NON_MARGINALITY]
        )


        print(json.dumps(report, indent=4))


def check_on_BT_problem():
    problem = EfficientBTProblem.from_default_files()
    search_space = ProblemSearchSpace([CombinatorialVariable(card) for card in problem.search_space.cardinalities])

    print(f"The cardinalities are {(problem.search_space.cardinalities)}")

    def fitness_function(sol):
        # wraps it into a Solution
        return problem.fitness_function(FullSolution(sol))

    report = detect_effects_on_dynamic_dataset(
        dynamic_fitness_function=DynamicFitnessFunctionWithRandomSample(fitness_function=fitness_function,
                                                                        search_space=search_space,
                                                                        quantity_of_samples=1000,
                                                                        to_maximise=True),
        which_variables=list(range(problem.search_space.amount_of_parameters)),
        which_effects=[AnnoyingEffects.NON_MARGINALITY]
    )

    print(json.dumps(report, indent=4))


#check_on_NK()


def make_linear_model(X_bool, y):
    encoder = OneHotEncoder(drop=None, dtype=int)
    X_encoded = encoder.fit_transform(X_bool)

    # Fit linear model
    model = LinearRegression()
    model.fit(X_encoded, y)
    y_pred = model.predict(X_encoded)

    # Print the model formula
    feature_names = encoder.get_feature_names_out()
    terms = [f"{coef:.3f} * {name}" for coef, name in zip(model.coef_, feature_names)]
    formula = " + ".join(terms) + f" + {model.intercept_:.3f}"
    print("Linear model:")
    print("y =", formula)

    # Print mean squared error
    mse = mean_squared_error(y, y_pred)
    print(f"\nMean Squared Error: {mse:.4f}")


def investigate_NK():
    problem = NK(n = 5, k = 2, fitness_functions=[[0.7478856972652854, 0.6711053869601792, 0.274728430820396, 0.7515953134478484, 0.9212612306042262, 0.21746404354376125, 0.3766682895444021, 0.09393150670032824],
[0.5699325498453233, 0.9468516435915038, 0.488511571462945, 0.4658184333986629, 0.9329105221737032, 0.7067307593865906, 0.26030310862860917, 0.24530654629684456],
[0.2541636289485518, 0.9746583084424283, 0.44202886852674705, 0.8596716878154749, 0.2820042608023454, 0.8102221039763946, 0.15781368947973062, 0.3709962972591462],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0]])

    def fitness_function(sol):
        # wraps it into a Solution
        return problem.fitness_function(FullSolution(sol))


    search_space = ProblemSearchSpace([CombinatorialVariable(card) for card in problem.search_space.cardinalities])
    solutions = search_space.enumerate()
    for solution in solutions:
        print("\t".join(map(repr, solution)) + f"\t{fitness_function(solution)}")

    make_linear_model(np.array(solutions), np.array([fitness_function(sol) for sol in solutions]))

    def fitness_function(sol):
        # wraps it into a Solution
        return problem.fitness_function(FullSolution(sol))

    report = detect_effects_on_dynamic_dataset(
        dynamic_fitness_function=DynamicFitnessFunctionWithRandomSample(fitness_function=fitness_function,
                                                                        search_space=search_space,
                                                                        quantity_of_samples=1000,
                                                                        to_maximise=True),
        which_variables=list(range(problem.search_space.amount_of_parameters)),
        which_effects=[AnnoyingEffects.NON_MARGINALITY]
    )

    print(json.dumps(report, indent=4))

check_on_BT_problem()

# investigate_NK()
