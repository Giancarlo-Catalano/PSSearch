import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import time

from pymoo.core.problem import ElementwiseProblem, Problem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination import get_termination
from pymoo.core.crossover import Crossover
from pymoo.optimize import minimize

from sklearn.model_selection import train_test_split, GridSearchCV


import warnings
warnings.filterwarnings("ignore")

class PS4RSProblem(Problem):

    def __init__(self, sessions, sessions_objective_values, stype=0):
        super().__init__(n_var=100,
                         n_obj=2,
                         xl=0,
                         xu=1,
                         )
        self.fitness  = sessions_objective_values
        self.sessions = sessions
        self.stype = stype

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = - mean_fitness(x, self.sessions, self.fitness, stype=self.stype).reshape(-1, 1)
        f2 = - simplicity(x).reshape(-1, 1)
        # f3 = - atomicity(x, self.sessions, self.fitness).reshape(-1, 1)

        out["F"] = [f1, f2]

