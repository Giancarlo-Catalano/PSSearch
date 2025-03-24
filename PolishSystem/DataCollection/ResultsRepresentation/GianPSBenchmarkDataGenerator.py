from typing import Optional

import numpy as np
from datasets import Dataset

import utils
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


class GianPSBenchmarkDataGenerator(BenchmarkDataGeneratorInterface):
    search_settings: PSSearchSettings

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

    @classmethod
    def get_metric_function(cls,
                            metric_name: str,
                            pRef: Optional[PRef] = None,
                            solution: Optional[FullSolution] = None,
                            problem: Optional[BenchmarkProblem] = None):

        # this is different from the one declared in ps_search_utils, because the signs do matter!

        def invert_if_required(func):
            if metric_name.startswith("-"):
                def with_inverted_sign(func):
                    def result_func(ps: PS):
                        return -func(ps)

                    return result_func

                return with_inverted_sign(func)
            else:
                return func

        actual_metric_name = metric_name[1:] if metric_name.startswith("-") else metric_name

        if actual_metric_name == "simplicity":
            def simplicity(ps: PS) -> float:
                return -float(np.sum(ps.values == STAR))

            return simplicity

        if actual_metric_name == "mean_fitness":
            mean_fitness_evaluator = MeanFitness()
            mean_fitness_evaluator.set_pRef(pRef)
            return invert_if_required(mean_fitness_evaluator.get_single_score)

        if actual_metric_name == "ground_truth_atomicity":
            ground_truth_atomicity_evaluator = TraditionalPerturbationLinkage(problem)
            assert(solution is not None)
            ground_truth_atomicity_evaluator.set_solution(solution)
            return invert_if_required(ground_truth_atomicity_evaluator.get_atomicity)

        if actual_metric_name == "estimated_atomicity":
            estimated_atomicity_metric = FasterSolutionSpecificMutualInformation()
            estimated_atomicity_metric.set_pRef(pRef)
            estimated_atomicity_metric.set_solution(solution)
            assert (solution is not None)

            return invert_if_required(estimated_atomicity_metric.get_atomicity)

        if actual_metric_name == "estimated_atomicity&evaluator":
            estimated_atomicity_metric = FasterSolutionSpecificMutualInformation()
            estimated_atomicity_metric.set_pRef(pRef)
            estimated_atomicity_metric.set_solution(solution)

            return estimated_atomicity_metric, invert_if_required(estimated_atomicity_metric.get_atomicity)

        if actual_metric_name == "consistency":
            fitness_consistency_evaluator = MannWhitneyU()
            fitness_consistency_evaluator.set_pRef(pRef)
            return invert_if_required(fitness_consistency_evaluator.get_single_score)

        if actual_metric_name == "variance":
            variance_evaluator = SplitVariance(pRef)
            return invert_if_required(variance_evaluator.get_single_score)

        if actual_metric_name == "sample_count":
            count_evaluator = SampleCount()
            count_evaluator.set_pRef(pRef)
            return invert_if_required(count_evaluator.get_single_score)

        raise NotImplemented

    def get_objectives(self, pRef: PRef):

        # based on search_settings.metrics
        return [self.get_metric_function(metric_str, pRef =pRef)
                for metric_str in self.search_settings.metrics.split()]

    def generate_pss(self) -> list[PS]:
        pRef = self.dataset.get_session_data()
        return search_global_polish_ps(original_problem_search_space=pRef.search_space,
                                       search_settings=self.search_settings,
                                       objectives=self.get_objectives(pRef))


    def to_klaudia_dict(self):
        individual_objectives = self.search_settings.metrics.split()

        return { "provided_by": str(self.user),
                 "date": utils.get_formatted_timestamp(),
                "input": self.dataset.to_dict(),
                "parameters": {
                    "n_gen": self.search_settings.ps_n_generations,
                    "pop_size": self.search_settings.ps_search_population,
                    "n_offsprings": self.search_settings.ps_search_population,
                    "crossover": self.search_settings.crossover_operator_name,
                    "initial_population": self.search_settings.sampling_operator_name,
                    "mutation": self.search_settings.mutation_operator_name,
                    "algorithm": "NSGA2" if len(individual_objectives) > 1 else "GA",
                    "objectives": individual_objectives
                },
                 "results": self.get_result_dict()
            }




