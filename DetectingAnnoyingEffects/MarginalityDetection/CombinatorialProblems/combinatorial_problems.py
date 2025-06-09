import numpy as np

from DetectingAnnoyingEffects.MarginalityDetection.CombinatorialProblems.CombinatorialProblem import \
    CombinatorialProblem


class UnitaryProblem(CombinatorialProblem):

    def __init__(self, n: int):
        cardinalities = np.ones(n, dtype=int) * 2
        super().__init__(cardinalities)

    def unitary_function(self, ones):
        raise NotImplementedError()

    def fitness_function(self, x):
        return self.unitary_function(np.sum(x))


class RR(UnitaryProblem):
    def unitary_function(self, ones):
        return float(self.n) if ones == self.n else 0.0


class Trap(UnitaryProblem):
    def unitary_function(self, ones):
        return self.n if ones == self.n else self.n - 1 - ones


class TwoPeaks(UnitaryProblem):
    def unitary_function(self, ones):
        return max(ones, self.n - ones)


class Parity(UnitaryProblem):
    def unitary_function(self, ones):
        return float(ones % 2) * self.n


class OneMax(UnitaryProblem):
    def unitary_function(self, ones):
        return ones
