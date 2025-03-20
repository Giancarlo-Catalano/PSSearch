import random
from typing import Any

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace
from PolishSystem.PolishPSSearchTask import LocalPSPolishSearchTask

Embedding = np.ndarray


class OnlyIfCloseEnoughCrossover(Crossover):
    # this is skeleton code to implement the crossover operator discussed on 13/3/25.

    pymoo_problem: LocalPSPolishSearchTask  # this is needed to convert the pymoo individuals into PSs
    embedding_model: Any  # this will be needed to convert PSs into embeddings
    distance_threshold: float

    def __init__(self,
                 pymoo_problem: LocalPSPolishSearchTask,
                 embedding_model: Any,
                 distance_threshold: float,
                 n_offsprings=2,
                 prob: float = 0.5,
                 **kwargs):
        self.pymoo_problem = pymoo_problem
        self.embedding_model = embedding_model
        self.distance_threshold = distance_threshold
        super().__init__(2, n_offsprings, prob=prob, **kwargs)

    def get_embedding_of_ps(self, mother_ps) -> Embedding:
        raise NotImplemented

    def get_distance_of_embeddings(self, mother_embedded, father_embedded) -> float:
        raise NotImplemented  # self.embedding_model.cosine_similarity(mother_embedded, father_embedded)

    def similarity_of_parents(self, mother_x, father_x) -> float:
        mother_ps = self.pymoo_problem.individual_to_ps(mother_x)
        father_ps = self.pymoo_problem.individual_to_ps(father_x)
        mother_embedded = self.get_embedding_of_ps(mother_ps)
        father_embedded = self.get_embedding_of_ps(father_ps)
        return self.get_distance_of_embeddings(mother_embedded, father_embedded)

    @classmethod
    def ps_uniform_crossover(self, mother: np.ndarray, father: np.ndarray):
        daughter = mother.copy()
        son = father.copy()

        swaps = np.random.random(len(mother)) < 0.5
        daughter[swaps], son[swaps] = son[swaps], daughter[swaps]

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape

        children = np.array([self.ps_uniform_crossover(mother, father)
                             for mother, father in zip(X[0], X[1])
                             if self.similarity_of_parents(mother, father) < self.distance_threshold])

        return np.swapaxes(children, 0, 1)
