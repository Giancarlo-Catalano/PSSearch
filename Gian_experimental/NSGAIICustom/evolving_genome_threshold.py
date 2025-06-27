import random
from typing import Iterable, Optional

import numpy as np

from Gian_experimental.NSGAIICustom.NSGAIICustom import NCSolution, NCSampler, NCMutation, NCCrossover


class NCSolutionWithGT(NCSolution):
    genome_threshold: Optional[int]

    def __init__(self, values: Iterable[int], genome_threshold: Optional[int]):
        super().__init__(values)
        self.genome_threshold = genome_threshold

    def __repr__(self):
        return f"{set(self)}, gt={self.genome_threshold}"

    @classmethod
    def without_gt(cls, solution: NCSolution):
        return cls(solution, genome_threshold=len(solution))

    def __eq__(self, other):
        return (super().__eq__(other)) and (self.genome_threshold == other.genome_threshold)


    def __hash__(self):
        return hash(sum(self) + (0 if self.genome_threshold is None else self.genome_threshold))


class SampleWithFixedGT(NCSampler):
    original_sampler: NCSampler
    genome_threshold: int

    def __init__(self, original_sampler: NCSampler, genome_threshold: int):
        self.original_sampler = original_sampler
        self.genome_threshold = genome_threshold
        super().__init__()

    def sample(self) -> NCSolution:
        return NCSolutionWithGT(self.original_sampler.sample(), genome_threshold=self.genome_threshold)


class AlsoSampleGT(NCSampler):
    original_sampler: NCSampler

    def __init__(self, original_sampler: NCSampler):
        self.original_sampler = original_sampler
        super().__init__()

    def sample_gt(self, solution: NCSolution) -> Optional[int]:
        raise NotImplementedError()

    def sample(self) -> NCSolution:
        solution = self.original_sampler.sample()
        produced = NCSolutionWithGT(solution, genome_threshold=self.sample_gt(solution))
        # if produced.genome_threshold is None:
        #     print(f"Impostor is in this operator! {type(self)}")
        return produced


class SimpleSampleGT(AlsoSampleGT):
    max_gt = 5

    def sample_gt(self, solution: NCSolution) -> Optional[int]:
        if len(solution) == 0:
            return 0  # but this shouldn't happen
        max_acceptable_gt = min(self.max_gt, len(solution))
        possible_values = list(range(1, max_acceptable_gt + 1))
        return random.choice(possible_values)


class MutateExceptGT(NCMutation):
    original_mutation: NCMutation

    def __init__(self, original_mutation: NCMutation):
        self.original_mutation = original_mutation
        super().__init__()

    def mutated(self, solution: NCSolutionWithGT) -> NCSolution:
        mutated_set = self.original_mutation.mutated(solution)
        return NCSolutionWithGT(mutated_set, genome_threshold=solution.genome_threshold)


class AlsoMutateGT(NCMutation):
    original_mutation: NCMutation

    def __init__(self, original_mutation: NCMutation):
        self.original_mutation = original_mutation
        super().__init__()

    def mutated_gt(self, original_solution: NCSolutionWithGT):
        raise NotImplementedError()

    def mutated(self, solution: NCSolutionWithGT) -> NCSolution:
        mutated_set = self.original_mutation.mutated(solution)
        produced = NCSolutionWithGT(mutated_set, genome_threshold=self.mutated_gt(solution))
        # if produced.genome_threshold is None:
        #     print(f"Impostor is in this operator! {type(self)}")
        #     raise NotImplementedError()
        return produced


class SimpleMutateGT(AlsoMutateGT):
    probability_of_change = 0.1
    max_gt = 5

    def mutated_gt(self, original_solution: NCSolutionWithGT):
        if random.random() < self.probability_of_change:
            if original_solution.genome_threshold is None:
                return random.choice(range(1, 6))
            new_genome_threshold = original_solution.genome_threshold + random.choice([+1, -1])
            return np.clip(new_genome_threshold, 1, self.max_gt)
        else:
            return original_solution.genome_threshold


class AlsoCrossoverGT(NCCrossover):
    original_crossover: NCCrossover

    def __init__(self, original_crossover: NCCrossover):
        self.original_crossover = original_crossover
        super().__init__()

    def crossed_gt(self, a: NCSolutionWithGT, b: NCSolutionWithGT) -> (int, int):
        raise NotImplementedError()

    def crossed(self, a: NCSolutionWithGT, b: NCSolutionWithGT) -> (NCSolution, NCSolution):
        child_1, child_2 = self.original_crossover.crossed(a, b)
        gt_1, gt_2 = self.crossed_gt(a, b)
        # if gt_1 is None or gt_2 is None:
        #     print(f"The impostor is here! {type(self)}")
        #     raise NotImplementedError()
        return NCSolutionWithGT(child_1, gt_1), NCSolutionWithGT(child_2, gt_2)


class SimpleCrossoverGT(AlsoCrossoverGT):
    crossover_chance: float = 0.5

    def crossed_gt(self, a: NCSolutionWithGT, b: NCSolutionWithGT) -> (int, int):
        if random.random() < self.crossover_chance:
            return b.genome_threshold, a.genome_threshold
        else:
            return a.genome_threshold, b.genome_threshold


def check():
    from Gian_experimental.NSGAIICustom.CustomOperators import NCSamplerFromPRef, NCMutationCounterproductive, \
        NCCrossoverTransition
    from Gian_experimental.NSGAIICustom.NSGAIICustom import EvaluatedNCSolution, NCSamplerSimple, NCMutationSimple, \
        NCCrossoverSimple
    from Gian_experimental.NSGAIICustom.testing_in_vitro.testing_operators_in_vitro import make_metrics_cached

    # dir_250 = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting\250"
    # dir_250 = r"/Users/gian/PycharmProjects/PSSearch/data/retail_forecasting/250"

    data_path = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting"
    import itertools

    import numpy as np

    from Gian_experimental.NSGAIICustom.testing_in_vitro.SPRef import OptimisedSPref
    from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import gian_get_similarities
    from PolishSystem.read_data import get_pRef_from_vectors
    import os
    from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import get_transition_matrix
    from typing import Iterable
    import heapq

    def in_path(path):
        return os.path.join(data_path, path)

    n = 250

    train_pRef = get_pRef_from_vectors(name_of_vectors_file=in_path(f"train_many_hot_vectors_{n}_random.csv"),
                                       name_of_fitness_file=in_path(f"train_fitness_{n}_random.csv"),
                                       column_in_fitness_file=2)
    test_pRef = get_pRef_from_vectors(name_of_vectors_file=in_path(f"test_many_hot_vectors_{n}_random.csv"),
                                      name_of_fitness_file=in_path(f"test_fitness_{n}_random.csv"),
                                      column_in_fitness_file=2)

    train_SPRef = OptimisedSPref.from_pRef(train_pRef)
    test_SPRef = OptimisedSPref.from_pRef(test_pRef)

    cluster_info_file_name = in_path(f"cluster_info_{n}_random.pkl")
    similarities = gian_get_similarities(cluster_info_file_name)

    def keep_ones_with_most_samples(population: Iterable[EvaluatedNCSolution], quantity_required: int):
        return heapq.nsmallest(iterable=population, key=lambda x: x.fitnesses[0], n=quantity_required)

    transition_matrix = get_transition_matrix(similarities)
    custom_sampling = NCSamplerFromPRef.from_PRef(train_pRef, allow_empty=False)
    custom_mutation = NCMutationCounterproductive(transition_matrix)
    custom_crossover = NCCrossoverTransition(transition_matrix)

    def atomicity(ps):
        if len(ps) < 2:
            return -1000
        else:
            linkages = [similarities[a, b] for a, b in itertools.combinations(ps, r=2)]
            return np.average(linkages)

    traditional_sampling = NCSamplerSimple.with_average_quantity(3, genome_size=n, allow_empty=False)
    traditional_mutation = NCMutationSimple(n)
    traditional_crossover = NCCrossoverSimple(swap_probability=1/n)

    # traditional_sampling_with_gt = SimpleSampleGT(traditional_sampling)
    # traditional_mutation_with_gt = SimpleMutateGT(traditional_mutation)
    # traditional_crossover_with_gt = SimpleCrossoverGT(traditional_crossover)

    traditional_sampling_with_gt = SampleWithFixedGT(traditional_sampling, genome_threshold=3)
    traditional_mutation_with_gt = MutateExceptGT(traditional_mutation)
    traditional_crossover_with_gt = SimpleCrossoverGT(traditional_crossover)

    from Gian_experimental.NSGAIICustom.NSGAIICustom import NSGAIICustom

    def get_metrics(ps: NCSolutionWithGT) -> tuple[float]:
        if ps.genome_threshold is None:
            print(f"Found the impostor!")
            raise NotImplementedError()

        matching, non_matching = train_SPRef.partition(ps, threshold=ps.genome_threshold)

        if len(matching) < 100:  # a constraint
            return 1000, 1000, 1000, 1000, 1000

        median_diff = np.median(matching) - np.median(non_matching)
        atomicity_score = atomicity(ps)

        return (len(ps), -len(matching), -median_diff, -atomicity_score)

    algorithm = NSGAIICustom(sampling=traditional_sampling_with_gt,
                             mutation=traditional_mutation_with_gt,
                             crossover=traditional_crossover_with_gt,
                             probability_of_crossover=0.5,
                             eval_budget=2000,
                             pop_size=100,
                             tournament_size=3,
                             mo_fitness_function=make_metrics_cached(get_metrics),
                             unique=True,
                             verbose=True,
                             culler=keep_ones_with_most_samples
                             )

    results = algorithm.run()

    for result in results:
        print(result)


#check()
