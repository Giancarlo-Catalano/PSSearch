import json
import random
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from anytree import RenderTree, Node

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from DecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor
from SimplifiedSystem.LocalPSSearchTask import find_ps_in_solution

import anytree

from SimplifiedSystem.PSSearchSettings import PSSearchSettings, get_default_search_settings


class PSDecisionTreeNode:
    prediction: float

    other_statistics: dict[str, float]
    fitnesses: Optional[np.ndarray]

    def __init__(self,
                 prediction: float,
                 other_statistics: dict[str, float],
                 fitnesses : Optional[np.ndarray] = None):
        self.prediction = prediction
        self.other_statistics = other_statistics
        self.fitnesses = fitnesses

    @classmethod
    def get_statistics_from_pRef(cls, pRef: PRef) -> dict[str, float]:
        fitnesses = pRef.fitness_array

        stats = dict()
        stats["n"] = len(fitnesses)

        if len(fitnesses) > 0:
            stats["average"] = np.average(fitnesses)

        if len(fitnesses) > 1:
            stats["variance"] = np.var(fitnesses)
            stats["sd"] = np.std(fitnesses)
            average = stats["average"]
            stats["mse"] = np.average((fitnesses - average) ** 2)
            stats["mae"] = np.average(np.abs(fitnesses - average))
            stats["min"] = np.min(fitnesses)
            stats["max"] = np.max(fitnesses)

        return stats

    def as_dict(self) -> dict:
        raise NotImplemented

    def get_prediction_dict(self) -> dict:
        prediction_dict = dict()
        prediction_dict["other_statistics"] = self.other_statistics
        prediction_dict["prediction"] = self.prediction
        return prediction_dict

    @classmethod
    def from_pRef(cls, pRef: PRef):
        stats = PSDecisionTreeLeafNode.get_statistics_from_pRef(pRef)
        prediction = stats.get("average", float('nan'))
        return cls(prediction=prediction, other_statistics=stats, fitnesses=pRef.fitness_array)

    def repr_custom(self, custom_ps_repr, custom_prop_repr) -> str:
        raise NotImplemented

    @classmethod
    def from_dict(cls, d: dict):
        raise NotImplemented


    def get_node_text(self, custom_repr_ps, custom_partition_repr) -> str:
        raise NotImplemented


class PSDecisionTreeLeafNode(PSDecisionTreeNode):
    def __init__(self,
                 prediction: float,
                 other_statistics: dict[str, float],
                 fitnesses: np.ndarray):
        super().__init__(prediction=prediction, other_statistics=other_statistics, fitnesses=fitnesses)

    def __repr__(self):
        return f"LeafNode(prediction = {self.prediction:.2f})"

    def repr_custom(self, custom_ps_repr: Callable, custom_prop_repr:Callable):
        return self.__repr__()

    def as_dict(self) -> dict:
        return {"node_type": "leaf"} | self.get_prediction_dict()

    @classmethod
    def from_dict(cls, d: dict):
        assert (d["node_type"] == "leaf")
        return cls(prediction=d["prediction"],
                   other_statistics=d["other_statistics"])

    def get_node_text(self, custom_ps_repr: Callable, custom_partition_repr: Callable):
        # ps repr is ignored
        return "(Leaf) "+ custom_partition_repr(self)


class PSDecisionTreeBranchNode(PSDecisionTreeNode):
    split_ps: Optional[PS]

    matching_branch: Optional
    not_matching_branch: Optional

    def __init__(self,
                 prediction: float,
                 other_statistics: dict[str, float],
                 fitnesses: np.ndarray):
        super().__init__(prediction=prediction, other_statistics=other_statistics, fitnesses=fitnesses)
        self.split_ps = None

        self.matching_branch = None
        self.not_matching_branch = None

    @classmethod
    def find_splitting_ps(cls,
                          search_settings: PSSearchSettings,
                          pRef: PRef) -> PS:
        best_solution = pRef.get_best_solution()

        ps_candidates = find_ps_in_solution(pRef=pRef,
                                            ps_budget=search_settings.ps_search_budget,
                                            culling_method=search_settings.culling_method,
                                            population_size=search_settings.ps_search_population,
                                            to_explain=best_solution,
                                            problem=search_settings.original_problem,
                                            metrics=search_settings.metrics,
                                            verbose=search_settings.verbose)

        return ps_candidates[0] #  the culling method should leave just one left in the array anyway

    def get_node_text(self, custom_ps_repr: Callable, custom_partition_repr: Callable):
        # ps repr is ignored
        result = ("(Branch) " +
                  utils.indent("\n"+custom_ps_repr(self.split_ps)) +
                  utils.indent("\n"+custom_partition_repr(self)))

        return result

    def repr_custom(self, custom_ps_repr: Callable, custom_prop_repr: Callable):
        result = self.get_node_text(custom_ps_repr, custom_prop_repr)

        result += (f",\n"
                   f"matching = \n"
                   f"{utils.indent(self.matching_branch.repr_custom(custom_ps_repr, custom_prop_repr))},\n"
                   f"not_matching = \n"
                   f"   {utils.indent(self.not_matching_branch.repr_custom(custom_ps_repr, custom_prop_repr))}")

        return result

    @classmethod
    def plain_ps_repr(cls, ps: PS) -> str:
        return " ".join("*" if value == STAR else f"{value}" for value in ps.values)


    def as_dict(self) -> dict:
        own_dict = {"node_type": "branch",
                    "split_ps": self.plain_ps_repr(self.split_ps),
                    "matching_branch": self.matching_branch.as_dict(),
                    "not_matching_branch": self.not_matching_branch.as_dict()}

        return own_dict | self.get_prediction_dict()

    @classmethod
    def get_node_from_dict(cls, d: dict) -> PSDecisionTreeNode:
        if d["node_type"] == "branch":
            return cls.from_dict(d)
        else:
            return PSDecisionTreeLeafNode.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict):
        assert (d["node_type"] == "branch")

        result_node = cls(prediction=d["prediction"],
                          other_statistics=d["other_statistics"])
        result_node.matching_branch = cls.get_node_from_dict(d["matching_branch"])
        result_node.not_matching_branch = cls.get_node_from_dict(d["not_matching_branch"])
        result_node.split_ps = PS(STAR if c == "*" else int(c) for c in d["split_ps"].split())
        return result_node



class PSDecisionTree(AbstractDecisionTreeRegressor):
    root_node: Optional[PSDecisionTreeNode]
    search_settings: Optional[PSSearchSettings]
    problem: Optional[BenchmarkProblem]

    def __init__(self,
                 maximum_depth: int,
                 search_settings: PSSearchSettings):
        self.root_node = None
        self.search_settings = search_settings


        super().__init__(maximum_depth=maximum_depth)

    def train_from_pRef(self, pRef: PRef, random_state: Optional[int] = None):
        random_state = random.randrange(10000) if random_state is None else random_state

        def recursively_train_node(pRef_to_split: PRef,
                                   current_depth: int) -> PSDecisionTreeNode:
            print(f"Splitting a pRef of size {pRef_to_split.sample_size}")
            if (current_depth >= self.maximum_depth) or (pRef_to_split.sample_size < 2):
                return PSDecisionTreeLeafNode.from_pRef(pRef_to_split)

            # otherwise we split more
            node = PSDecisionTreeBranchNode.from_pRef(pRef_to_split)
            splitting_ps = PSDecisionTreeBranchNode.find_splitting_ps(search_settings=self.search_settings,
                                                                      pRef=pRef_to_split)

            if self.search_settings.verbose:
                print(f"The splitting PS is {splitting_ps}")
            node.split_ps = splitting_ps
            matching_indexes = pRef_to_split.get_indexes_matching_ps(splitting_ps)
            matching_pRef, not_matching_pRef = pRef_to_split.split_by_indexes(matching_indexes)


            node.matching_branch = recursively_train_node(pRef_to_split=matching_pRef,
                                                          current_depth=current_depth + 1)

            node.not_matching_branch = recursively_train_node(pRef_to_split=not_matching_pRef,
                                                              current_depth=current_depth + 1)
            node.matching_branch.fitnesses = matching_pRef.fitness_array
            node.not_matching_branch.fitnesses = not_matching_pRef.fitness_array

            return node

        self.root_node = recursively_train_node(pRef_to_split=pRef,
                                                current_depth=0)

    def get_prediction(self, solution: FullSolution) -> float:

        def recursive_prediction(current_node: PSDecisionTreeNode) -> float:
            if isinstance(current_node, PSDecisionTreeLeafNode):
                return current_node.prediction
            elif isinstance(current_node, PSDecisionTreeBranchNode):
                if contains(solution, current_node.split_ps):
                    return recursive_prediction(current_node.matching_branch)
                else:
                    return recursive_prediction(current_node.not_matching_branch)

        return recursive_prediction(self.root_node)

    def all_nodes_as_list(self) -> list[PSDecisionTreeNode]:

        accumulator = []

        def recursively_register_node(current_node: PSDecisionTreeNode) -> None:
            accumulator.append(current_node)
            if isinstance(current_node, PSDecisionTreeBranchNode):
                recursively_register_node(current_node.matching_branch)
                recursively_register_node(current_node.not_matching_branch)

        recursively_register_node(self.root_node)
        return accumulator

    def all_pss_as_list(self) -> list[PS]:
        return [node.split_ps for node in self.all_nodes_as_list() if isinstance(node, PSDecisionTreeBranchNode)]


    def __repr__(self):
        return f"PS Decision tree of max depth {self.maximum_depth}"

    def as_dict(self) -> dict:
        result = {"maximum_depth": self.maximum_depth}
        if self.search_settings is not None:
            result["search_settings"] = self.search_settings.as_dict()

        if self.root_node is not None:
            result["tree"] = self.root_node.as_dict()

        return result

    @classmethod
    def from_dict(cls, d: dict):
        result = cls(maximum_depth=d["maximum_depth"], search_settings=get_default_search_settings())
        result.root_node = PSDecisionTreeBranchNode.get_node_from_dict(d["tree"]) if "tree" in d else None
        return result

    @classmethod
    def from_file(cls, filename: str):
        with open(filename, "r") as file:
            data = json.load(file)
        return cls.from_dict(data)

    def to_file(self, filename: str):
        with utils.open_and_make_directories(filename) as file:
            data = self.as_dict()
            json.dump(data, file, indent=4)

    def with_permutation(self, permutation: list[int]):
        def permute_ps(ps: PS) -> PS:
            return PS(ps.values[permutation])

        def permute_node(node: PSDecisionTreeNode) -> PSDecisionTreeNode:
            if isinstance(node, PSDecisionTreeBranchNode):
                result = PSDecisionTreeBranchNode(prediction=node.prediction,
                                                  other_statistics=node.other_statistics)
                result.split_ps = permute_ps(node.split_ps)
                result.matching_branch = permute_node(node.matching_branch)
                result.not_matching_branch = permute_node(node.not_matching_branch)
                return result
            else:
                return node

        result = PSDecisionTree(maximum_depth=self.maximum_depth)
        result.root_node = permute_node(self.root_node)
        result.search_settings = self.search_settings
        result.problem = None  # this needs to be set somewhere else, since the problem will be different
        return result


    def print_ASCII(self,
                    show_not_matching_nodes: bool = True,
                    custom_ps_repr: Optional[Callable] = None,
                    custom_partition_repr: Optional[Callable] = None):

        ps_repr = repr if custom_ps_repr is None else custom_ps_repr

        def default_partition_repr(node: PSDecisionTreeNode):
            return "stats:"+", ".join(f"{stat}->{value}" for stat, value in node.other_statistics.items())

        partition_repr = default_partition_repr if custom_partition_repr is None else custom_partition_repr
        def add_node_repr(node: PSDecisionTreeNode, parent, preamble: str):
            own_node_repr = Node(preamble+"\n"+node.get_node_text(ps_repr, partition_repr), parent = parent)
            if isinstance(node, PSDecisionTreeBranchNode):
                add_node_repr(node.matching_branch, parent = own_node_repr, preamble = "Matching")
                if show_not_matching_nodes:
                    add_node_repr(node.not_matching_branch, parent = own_node_repr, preamble = "NOT matching")
            return own_node_repr

        root_node_repr = add_node_repr(self.root_node, parent = None, preamble="Root")
        for pre, fill, node in RenderTree(root_node_repr):
            lines = node.name.splitlines()
            # Print the first line with the usual prefix.
            print(f"{pre}{lines[0]}")
            # For any additional lines, print them with an indentation that matches the node's position.
            for line in lines[1:]:
                print(f"{fill}{line}")
