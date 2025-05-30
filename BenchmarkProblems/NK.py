import random

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace


class NK(BenchmarkProblem):
    n: int  # genome size
    k: int  # k+1 is the size of the cliques

    fitness_functions: list[list[float]]

    def __init__(self, n, k, fitness_functions: list[list[float]]):
        super().__init__(SearchSpace([2 for _ in range(n)]))
        self.n = n
        self.k = k
        self.fitness_functions = fitness_functions

    @classmethod
    def random(cls, n: int, k: int):
        possible_combinations_for_each_clique = 2 ** (k + 1)
        quantity_of_subfunctions = n

        def get_random_fitness():
            return random.uniform(0, 1)  # floats are precise, so I'm happy with this

        def get_random_subfunction():
            return [get_random_fitness() for _ in range(possible_combinations_for_each_clique)]

        subfunctions = [get_random_subfunction() for _ in range(quantity_of_subfunctions)]
        return cls(n, k, subfunctions)

    def fitness_function(self, fs: FullSolution) -> float:
        extended_values = np.hstack((fs.values, fs.values[:self.k]))
        powers_of_2 = 2 ** np.arange(self.k + 1)
        clique_values = np.convolve(extended_values, powers_of_2, "valid")
        return sum(f_i[c_i] for f_i, c_i in zip(self.fitness_functions, clique_values))



def check_nk():
    problem_no_overlap = NK.random(6, 0)
    problem_1_overlap = NK.random(6, 1)
    problem_all_overlap = NK.random(6, 6)


    solutions = [FullSolution.random(problem_no_overlap.search_space) for _ in range(12)]

    for problem in [problem_no_overlap, problem_1_overlap, problem_all_overlap]:
        for solution in solutions:
            print(solution)
            fitness = problem.fitness_function(solution)
            print(fitness)


#check_nk()