from typing import Optional, Literal, Callable, TypeAlias

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from SimplifiedSystem.Operators.Sampling import LocalPSGeometricSampling
from SimplifiedSystem.ps_search_utils import construct_objectives_list, apply_culling_method, \
    run_pymoo_algorithm_with_checks
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR

PSObjective: TypeAlias = Callable[[PS], float]


class LocalPSSearchTask(Problem):
    solution_to_explain: FullSolution
    unexplained_mask: np.ndarray
    proportion_unexplained_that_needs_used: float  # alpha
    proportion_used_that_should_be_unexplained: float  # beta

    objectives: list[Callable]
    difference_variables: list[int]

    # NOTE:
    #  if you want the PS to completely ignore the already explained stuff, set beta to 1
    #  if you want the unexplained stuff to be completely contained in the PS, set alpha to 1
    # if alpha = 0.5, beta = 0.5, then
    # at least half of the PS is new stuff
    # at least half of the new stuff is in the PS
    # generally, you would want at least one new thing to be used, set alpha to > 0
    # generally, you would want most of the PS to contain new things, set beta to > 0.5

    def __init__(self,
                 solution_to_explain: FullSolution,
                 objectives: list[Callable],
                 unexplained_mask: Optional[np.ndarray] = None,
                 proportion_unexplained_that_needs_used: float = 0.01,  # at least
                 proportion_used_that_should_be_unexplained: float = 0.5):  # at least
        self.solution_to_explain = solution_to_explain
        self.objectives = objectives
        self.unexplained_mask = np.ones(shape=len(solution_to_explain),
                                        dtype=bool) if unexplained_mask is None else unexplained_mask
        self.difference_variables = np.arange(len(self.unexplained_mask))[self.unexplained_mask]  # gets the indexes

        self.proportion_unexplained_that_needs_used = proportion_unexplained_that_needs_used
        self.proportion_used_that_should_be_unexplained = proportion_used_that_should_be_unexplained

        # then the stuff to satisfy pymoo
        n_var = len(solution_to_explain.values)
        lower_bounds = np.full(shape=n_var, fill_value=0)  # the stars
        upper_bounds = lower_bounds + 1
        super().__init__(n_var=n_var,
                         n_obj=len(self.objectives),
                         n_ieq_constr=1,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         vtype=bool)

    def individual_to_ps(self, x):
        return PS(sol_value if x_value == 1 else -1 for (sol_value, x_value) in zip(self.solution_to_explain.values, x))

    def get_which_rows_satisfy_constraint(self, X: np.ndarray) -> np.ndarray:
        # for a ps with some fixed variables, there are
        #  F which are used in the PS
        #  U is the amount of unexplained variables
        #  H which are unexplained and used in the PS
        #  (a) (H/F)% is how many of the used variables are unexplained, should be greater than proportion alpha
        # -> H / F >= alpha <=> H >= F * alpha
        #  (b) (H/U)% is how many of the unexplained variables are used, should be greater than proportion beta
        # -> H / U >= beta <=> H >= U * beta

        f = np.sum(X, axis=1)
        u = len(self.difference_variables)
        h = np.sum(X[:, self.difference_variables], axis=1)

        threshold_h_A = f * self.proportion_used_that_should_be_unexplained
        threshold_h_B = u * self.proportion_unexplained_that_needs_used

        satisfies_A = h >= threshold_h_A
        satisfies_B = h >= threshold_h_B

        return np.logical_and(satisfies_A, satisfies_B)

    def get_metrics_for_ps(self, ps: PS) -> list[float]:
        return [objective(ps) for objective in self.objectives]

    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.get_metrics_for_ps(self.individual_to_ps(row)) for row in X])
        out["F"] = metrics

        out["G"] = 0.5 - self.get_which_rows_satisfy_constraint(
            X)  # if the constraint is satisfied, it is negative (which is counterintuitive)


def find_ps_in_solution(to_explain: FullSolution,
                        pRef: PRef,
                        ps_budget: int,
                        population_size: int = 100,
                        proportion_unexplained_that_needs_used: float = 0.01,
                        proportion_used_that_should_be_unexplained: float = 0.5,
                        culling_method=Optional[Literal["biggest", "least_dependent", "overlap"]],
                        reattempts_when_fail: int = 1,
                        unexplained_mask: Optional[np.ndarray] = None,
                        problem: Optional[BenchmarkProblem] = None,
                        metrics: str = "simplicity mean_fitness estimated_atomicity",
                        sampling_operator: Optional[Sampling] = None,
                        mutation_operator: Optional[Mutation] = None,
                        crossover_operator: Optional[Crossover] = None,
                        verbose=True) -> list[PS]:
    objectives = construct_objectives_list(metrics, pRef, to_explain, problem)

    if len(objectives) == 0:
        raise Exception("Somehow there are no objectives")

    # construct the optimisation problem instance
    problem = LocalPSSearchTask(solution_to_explain=to_explain,
                                objectives=objectives,
                                unexplained_mask=unexplained_mask,
                                proportion_unexplained_that_needs_used=proportion_unexplained_that_needs_used,
                                proportion_used_that_should_be_unexplained=proportion_used_that_should_be_unexplained)

    # if there are no operators given, we have these defaults
    sampling_operator = LocalPSGeometricSampling() if sampling_operator is None else sampling_operator
    crossover_operator = SimulatedBinaryCrossover(prob=0.3) if crossover_operator is None else crossover_operator
    mutation_operator = BitflipMutation(prob=1 / problem.n_var) if mutation_operator is None else mutation_operator

    # the next line of code is a bit odd, but it works! It uses a GA if there is one objective
    algorithm = (GA if len(objectives) < 2 else NSGA2)(pop_size=population_size,
                                                       sampling=sampling_operator,
                                                       crossover=crossover_operator,
                                                       mutation=mutation_operator,
                                                       eliminate_duplicates=True)

    pss = run_pymoo_algorithm_with_checks(pymoo_problem=problem,
                                          algorithm=algorithm,
                                          reattempts_when_fail=reattempts_when_fail,
                                          ps_budget = ps_budget,
                                          verbose=verbose)

    return apply_culling_method(pss, culling_method)
