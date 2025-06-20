from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace


class TwoPeaks(BenchmarkProblem):
    n: int


    def __init__(self, n: int):
        self.n = n
        super().__init__(SearchSpace([2 for _ in range(n)]))

    def fitness_function(self, fs: FullSolution) -> float:
        leading_ones = 0
        for value in fs.values:
            if value == 1:
                leading_ones +=1
            else:
                break
        trailing_zeros = 0
        for value in reversed(fs.values):
            if value == 0:
                trailing_zeros += 1
            else:
                break


        return max(leading_ones, trailing_zeros)