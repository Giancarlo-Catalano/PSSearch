from dataclasses import dataclass
from typing import Optional

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem


@dataclass
class PSSearchSettings:
    ps_search_budget: int
    ps_search_population: int

    metrics: str

    culling_method: str
    original_problem: Optional[BenchmarkProblem]
    verbose: bool

    proportion_unexplained_that_needs_used: float
    proportion_used_that_should_be_unexplained: float

    ps_n_generations: Optional[int] = None
    crossover_operator_name: Optional[str] = None
    mutation_operator_name: Optional[str] = None
    sampling_operator_name: Optional[str] = None

    crossover_operator: Optional[Crossover] = None
    mutation_operator: Optional[Mutation] = None
    sampling_operator: Optional[Sampling] = None

    def as_dict(self):

        return {"ps_search_budget": self.ps_search_budget,
                "ps_search_population": self.ps_search_population,
                "ps_n_generations": self.ps_n_generations,
                "metrics": self.metrics,
                "culling_method": self.culling_method,
                "proportion_unexplained_that_needs_used": self.proportion_unexplained_that_needs_used,
                "proportion_used_that_should_be_unexplained": self.proportion_used_that_should_be_unexplained,
                "crossover": self.crossover_operator_name,
                "mutation": self.mutation_operator_name,
                "sampling": self.sampling_operator_name,
                "verbose": self.verbose}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(ps_search_budget=d["ps_search_budget"],
                   ps_search_population=d["ps_search_population"],
                   ps_n_generations=d["ps_n_generations"],
                   metrics=d["metrics"],
                   culling_method=d["culling_method"],
                   original_problem=None,
                   proportion_used_that_should_be_unexplained=d["proportion_used_that_should_be_unexplained"],
                   proportion_unexplained_that_needs_used=d["proportion_unexplained_that_needs_used"],
                   crossover_operator_name=d["crossover_operator_name"],
                   mutation_operator_name=d["mutation_operator_name"],
                   sampling_operator_name=d["sampling_operator_name"],
                   verbose=d["verbose"])


def get_default_search_settings() -> PSSearchSettings:
    return PSSearchSettings(ps_search_budget=5000,
                            ps_search_population=100,
                            metrics="simplicity mean_fitness estimated_atomicity",
                            culling_method="biggest",
                            original_problem=None,
                            proportion_used_that_should_be_unexplained=0.5,
                            proportion_unexplained_that_needs_used=0.01,
                            verbose=True,
                            )
