# refer to https://github.com/CIRGLaboratory/ECXAI-2025-Stirling/blob/klaudia/pattern_analysis/2025-03-21-KBa-example-data-for-eval.ipynb
import os
# {
#     "provided_by": "KBa",
#     "date": "2025-03-21",
#     "input": "hierarchical_qmc\\train_many_hot_vectors_100_qmc.csv",
#     "parameters": {
#         "n_gen": 100,
#         "pop_size": 125,
#         "n_offsprings": 150,
#         "crossover": "FitnessCrossover",
#         "initial_population": "SequenceBasedSampling",
#         "mutation": "SingleNegativeMutation",
#         "algorithm": "NSGA2",
#         "objectives": [
#             "-mean_fitness",
#             "-simplicity"
#         ]
#     }
# }


from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import utils
from Core.PRef import PRef
from Core.PS import PS, STAR
from PolishSystem.read_data import get_pRef_from_vectors


class User(Enum):
    GIAN = 1
    KLAUDIA = 2

    def __str__(self):
        return ["dummy", "GC", "KBa"][self.value]

    def get_data_path(self):
        if self == User.GIAN:
            return r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting"
        elif self == User.KLAUDIA:
            return r"hierarchical_qmc"

    def to_dict(self):
        return {"user": str(self)}

    @classmethod
    def from_dict(cls, input_str: dict):
        match input_str:
            case "GC":
                return cls.GIAN
            case "KBa":
                return cls.KLAUDIA
            case _:
                raise NotImplemented


@dataclass
class DataSet:
    vector_size: int
    clustering_method: str
    fitness_column: int

    which: Literal["train", "test", "general"]

    user: User

    def get_session_file_name(self) -> str:
        result = f"many_hot_vectors_{self.vector_size}_{self.clustering_method}.csv"
        if self.which in {"train", "test"}:
            result = self.which + "_" + result
        return result

    def get_fitness_file_name(self) -> str:
        result = f"fitness_{self.vector_size}_{self.clustering_method}.csv"
        if self.which in {"train", "test"}:
            result = self.which + "_" + result
        return result

    def get_cluster_IDs_file_name(self) -> str:
        return f"cluster_IDs_{self.vector_size}_{self.clustering_method}.pkl"

    def get_cluster_info_file_name(self) -> str:
        return f"cluster_info_{self.vector_size}_{self.clustering_method}.pkl"

    def get_default_session_data_path(self) -> str:
        return os.path.join(self.user.get_data_path(), self.get_session_file_name())

    def get_default_fitness_data_path(self) -> str:
        return os.path.join(self.user.get_data_path(), self.get_fitness_file_name())

    def get_default_cluster_info_path(self) -> str:
        return os.path.join(self.user.get_data_path(), self.get_cluster_info_file_name())

    def get_default_cluster_IDs_path(self) -> str:
        return os.path.join(self.user.get_data_path(), self.get_cluster_IDs_file_name())

    def get_session_data(self,
                         session_data_path: Optional[str] = None,
                         fitness_data_path: Optional[str] = None,
                         fitness_column: int = None
                         ) -> PRef:
        session_data_path = self.get_default_session_data_path() if session_data_path is None else session_data_path
        fitness_data_path = self.get_default_fitness_data_path() if fitness_data_path is None else fitness_data_path

        return get_pRef_from_vectors(name_of_vectors_file=session_data_path,
                                     name_of_fitness_file=fitness_data_path,
                                     column_in_fitness_file=fitness_column)

    def to_dict(self):
        return ({"vector_size": self.vector_size,
                 "clustering_method": self.clustering_method,
                 "fitness_column": self.fitness_column,
                 "which": self.which}
                | self.user.to_dict())

    @classmethod
    def from_dict(cls, input_dict: dict):
        return cls(vector_size=input_dict["vector_size"],
                   clustering_method=input_dict["clustering_method"],
                   fitness_column=input_dict["fitness_column"],
                   user=User.from_dict(input_dict["user"]),
                   which=input_dict["which"])


@dataclass
class BenchmarkDataGeneratorInterface:
    seed: Optional[int]
    dataset: DataSet
    user: User
    results: Optional[list[PS]] = None

    def generate_pss(self) -> list[PS]:
        raise NotImplemented

    def run(self):
        with utils.announce("Running the benchmark"):
            self.results = self.generate_pss()

    def get_result_dict(self):
        def ps_to_list(ps: PS):
            return [value == 1 for value in ps.values]

        return {"results": list(map(ps_to_list, self.results))} if self.results is not None else {
            "results": None}
    def to_dict(self):
        return {"seed": self.seed} | self.dataset.to_dict() | self.user.to_dict() | self.get_result_dict()

    @classmethod
    def load_from_dict(cls, input_dict):
        def ps_from_list(input_list: list[bool]):
            return PS(1 if value else STAR for value in input_list)

        return cls(seed=input_dict["seed"],
                   dataset=input_dict["dataset"],
                   user=User.from_dict(input_dict["user"]),
                   results=list(map(ps_from_list, input_dict[""])))
