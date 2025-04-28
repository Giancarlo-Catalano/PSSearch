import itertools
from typing import Callable, Optional

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Core.PSMetric.Metric import Metric
from Core.SearchSpace import SearchSpace


class TraditionalPerturbationLinkage(Metric):
    optimisation_problem: BenchmarkProblem

    current_solution: Optional[EvaluatedFS]
    linkage_table: Optional[np.ndarray]

    def __init__(self, optimisation_problem: BenchmarkProblem):
        self.optimisation_problem = optimisation_problem
        self.current_solution = None
        self.linkage_table = None
        super().__init__()

    def set_solution(self, new_solution: FullSolution) -> None:
        if (self.current_solution is not None) and (self.current_solution == new_solution):
            return

        fitness = self.optimisation_problem.fitness_function(new_solution)
        self.linkage_table = self.generate_linkage_table(EvaluatedFS(new_solution, fitness))



    @property
    def cardinalities(self) -> np.ndarray:
        return self.optimisation_problem.search_space.cardinalities

    @property
    def n(self) -> int:
        return self.optimisation_problem.search_space.amount_of_parameters

    def generate_linkage_table(self, solution: EvaluatedFS):

        own_fitness = solution.fitness

        changes_dict = dict()

        ## univariate_changes
        for var, cardinality in enumerate(self.cardinalities):
            for new_value in range(cardinality):
                new_solution = solution.with_different_value(var, new_value)
                fitness = self.optimisation_problem.fitness_function(new_solution)
                changes_dict[(var, new_value)] = fitness

        # bivariate_changes
        for var_a, var_b in itertools.combinations(range(self.n), r=2):
            for val_a in range(self.cardinalities[var_a]):
                new_solution = solution.with_different_value(var_a, val_a)
                for val_b in range(self.cardinalities[var_b]):
                    new_solution = new_solution.with_different_value(var_b, val_b)
                    fitness = self.optimisation_problem.fitness_function(new_solution)
                    changes_dict[(var_a, val_a, var_b, val_b)] = fitness

        def get_linkage_between_vals(var_a, val_a, var_b, val_b) -> float:
            both = changes_dict[(var_a, val_a, var_b, val_b)]
            just_a = changes_dict[(var_a, val_a)]
            just_b = changes_dict[(var_b, val_b)]

            return abs(own_fitness + both - just_a - just_b)

        def get_linkage_between_vars(var_a, var_b) -> float:
            own_val_a = solution.values[var_a]
            own_val_b = solution.values[var_b]

            all_linkages = [get_linkage_between_vals(var_a, val_a, var_b, val_b)
                            for val_a in range(self.cardinalities[var_a])
                            for val_b in range(self.cardinalities[var_b])
                            if val_a != own_val_a
                            if val_b != own_val_b]
            if len(all_linkages) == 0:
                return 0

            return np.average(all_linkages)

        def get_importance_of_var(var_a: int) -> float:
            own_val_a = solution.values[var_a]
            other_fitnesses = [changes_dict[(var_a, val_a)]
                               for val_a in range(self.cardinalities[var_a])
                               if val_a != own_val_a]
            if len(other_fitnesses) == 0:
                return 0

            return np.average([abs(other_fit - own_fitness) for other_fit in other_fitnesses])

        linkage_table = np.zeros(shape=(self.n, self.n))
        for var_a, var_b in itertools.combinations(range(self.n), r=2):
            linkage_table[var_a, var_b] = get_linkage_between_vars(var_a, var_b)

        linkage_table += linkage_table.T

        for var_a in range(self.n):
            linkage_table[var_a, var_a] = get_importance_of_var(var_a)

        return linkage_table

    def get_atomicity(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()

        def weakest_internal_linkage_for(var) -> float:
            return min(self.linkage_table[var, other] for other in fixed_vars if other != var)

        if len(fixed_vars) > 1:
            weakest_links = np.array([weakest_internal_linkage_for(var) for var in fixed_vars])
            return np.average(weakest_links)
        elif len(fixed_vars) == 1:
            var = fixed_vars[0]
            return self.linkage_table[var, var]
        else:
            return 0

    def get_dependence(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()
        unfixed_vars = [index for index, val in enumerate(ps.values) if val == STAR]

        def strongest_external_linkage_for(var) -> float:
            return max(self.linkage_table[var, other] for other in unfixed_vars)

        if (len(unfixed_vars) > 0) and (len(fixed_vars) > 0):  # maybe this should be zero?
            strongest_links = np.array([strongest_external_linkage_for(var) for var in fixed_vars])
            return np.average(strongest_links)
        elif len(fixed_vars) == 1:
            var = fixed_vars[0]
            return strongest_external_linkage_for(var)
        else:
            return 0
    def get_table_for_ps(self, ps: PS) -> np.ndarray:
        fixed_vars = ps.get_fixed_variable_positions()
        return self.linkage_table[fixed_vars, :][:, fixed_vars]
