import numpy as np

from BenchmarkProblems.BT.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from DetectingAnnoyingEffects.MarginalityDetection.CombinatorialProblems.CombinatorialProblem import \
    CombinedCombinatorialProblem, CombinatorialProblem
from DetectingAnnoyingEffects.MarginalityDetection.CombinatorialProblems.combinatorial_problems import RR, Trap, Parity, \
    OneMax, TwoPeaks
from DetectingAnnoyingEffects.MarginalityDetection.CombinatorialProblems.detect_marginality_of_combinatorial import \
    detect_marginality_of_combinatorial


def run_on_toy_problems():
    amalgam_problem = CombinedCombinatorialProblem([RR(4), Trap(4), Parity(4), OneMax(4)], aggregation_func=sum)
    random_samples_quantity = 100
    random_samples = [amalgam_problem.random_solution() for _ in range(random_samples_quantity)]

    marginalities = detect_marginality_of_combinatorial(amalgam_problem, samples_to_test_on=random_samples)
    print(np.array(marginalities * 100, dtype=int))



class ConvertedProblem(CombinatorialProblem):
    original_problem: BenchmarkProblem


    def __init__(self, original_problem: BenchmarkProblem):
        super().__init__(original_problem.search_space.cardinalities)
        self.original_problem = original_problem

    def fitness_function(self, x):
        # wrap in a FullSolution
        return self.original_problem.fitness_function(FullSolution(x))
def run_on_BT_problem():
    original_problem = EfficientBTProblem.from_default_files()
    problem = ConvertedProblem(original_problem)
    #problem = CombinedCombinatorialProblem([OneMax(4), RR(4), TwoPeaks(4), Trap(4), Parity(4)], aggregation_func=sum)
    random_samples_quantity = 1000
    random_samples = [problem.random_solution() for _ in range(random_samples_quantity)]

    marginalities = detect_marginality_of_combinatorial(problem, samples_to_test_on=random_samples)
    print(marginalities)
    for var, cardinality, marginality in zip(range(1000), problem.cardinalities, marginalities):
        normalised = marginality / (cardinality * (cardinality -1))
        print(f"{var = }, {cardinality =}, {marginality = :.3f}, {normalised = :.3f}")


run_on_BT_problem()
