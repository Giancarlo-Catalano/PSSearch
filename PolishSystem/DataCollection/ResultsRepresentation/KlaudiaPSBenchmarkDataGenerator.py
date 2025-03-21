from typing import Optional

import numpy as np
from datasets import Dataset

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.MeanFitness import MeanFitness
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import MannWhitneyU
from Core.PSMetric.FitnessQuality.SplitVariance import SplitVariance
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from Core.PSMetric.Linkage.ValueSpecificMutualInformation import FasterSolutionSpecificMutualInformation
from Core.PSMetric.SampleCount import SampleCount
from PolishSystem.DataCollection.ResultsRepresentation.BenchmarkDataGeneratorInterface import \
    BenchmarkDataGeneratorInterface, User
from PolishSystem.polish_search_methods import search_global_polish_ps
from SimplifiedSystem.PSSearchSettings import PSSearchSettings

class KlaudiaPSBenchmarkDataGenerator(BenchmarkDataGeneratorInterface):
    pop_size: int
    n_gen: int
    n_offsprings: int

    sampling_operator_name: str = "SequenceBasedSampling(sessions, 0.35)"
    mutation_operator_name: str = "SingleNegativeMutation()"
    crossover_operator_name: str = "FitnessCrossover(..., n_offsprings=5)"

    objectives = ['-mean_fitness', '-simplicity']



    def __init__(self,
                 user: User,
                 search_settings: PSSearchSettings,
                 dataset: Dataset,
                 seed: Optional[int] = None,
                 ):
        self.search_settings = search_settings
        self.search_settings.culling_method = None  # just so we get all the PSs
        assert (self.search_settings.metrics is not None)
        super().__init__(user=user,
                         dataset=dataset,
                         seed=seed)

    def get_objectives(self):
        # based on search_settings.metrics
        return list(map(self.get_metric_function, self.search_settings.metrics.split()))

    def generate_pss(self) -> list[PS]:
        # this stuff does not work well...
        initial_population = SequenceBasedSampling(sessions, 0.35)
        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=initial_population,
            crossover=FitnessCrossover(fitness=sessions_objective_values[:, stype], n_offsprings=5),  # all basics work
            mutation=SingleNegativeMutation(),  # all basics work
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", n_gen)

        results = []

        for _ in range(10):
            res = minimize(problem,
                           algorithm,
                           termination,
                           save_history=True,
                           verbose=True)

            results.append(res.X)

        def X_row_to_PS(row: np.ndarray):
            return PS(1 if value else STAR for value in row)

        results_pss = [X_row_to_PS(row)
                       for X in results
                       for row in X]