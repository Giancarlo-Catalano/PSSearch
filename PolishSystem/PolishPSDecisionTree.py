import random
from typing import Callable
from typing import Optional

from Core.PRef import PRef
from Core.PS import PS
from DecisionTree.PSDecisionTree import PSDecisionTree, PSDecisionTreeNode, PSDecisionTreeLeafNode, \
    PSDecisionTreeBranchNode
from PolishSystem.polish_search_methods import search_global_polish_ps
from SimplifiedSystem.PSSearchSettings import PSSearchSettings


class PolishPSDecisionTree(PSDecisionTree):
    get_objectives_for_partition: Callable[[PRef], list[Callable[[PS], float]]]

    def __init__(self,
                 search_settings: PSSearchSettings,
                 maximum_depth: int,
                 get_objectives_for_partition: Callable):
        self.get_objectives_for_partition = get_objectives_for_partition
        super().__init__(search_settings=search_settings, maximum_depth=maximum_depth)

    def train_from_pRef(self, pRef: PRef, random_state: Optional[int] = None):
        random_state = random.randrange(10000) if random_state is None else random_state

        def recursively_train_node(pRef_to_split: PRef,
                                   current_depth: int) -> PSDecisionTreeNode:
            print(f"Splitting a pRef of size {pRef_to_split.sample_size}")
            if (current_depth >= self.maximum_depth) or (pRef_to_split.sample_size < 2):
                return PSDecisionTreeLeafNode.from_pRef(pRef_to_split)

            # otherwise we split more
            node = PSDecisionTreeBranchNode.from_pRef(pRef_to_split)

            # this is where we search for the PS
            objectives = self.get_objectives_for_partition(pRef_to_split)
            splitting_pss = search_global_polish_ps(original_problem_search_space=pRef_to_split.search_space,
                                                    search_settings=self.search_settings,
                                                    objectives=objectives)

            splitting_ps = splitting_pss[0]

            if self.search_settings.verbose:
                print(f"The splitting PS is {splitting_ps}")
                # print("The other ones that were found were")
                # for ps in splitting_pss:
                #     print(f"\t{ps}")
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
