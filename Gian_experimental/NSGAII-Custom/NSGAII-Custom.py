import itertools
import random
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

import utils
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from Core.get_pRef import get_pRef_from_metaheuristic
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import scale_to_have_sum_and_max, scale_to_have_sum

NCSolution = set[int]


class EvaluatedNCSolution:
    solution: NCSolution
    fitnesses: tuple[float]

    def __init__(self, solution, fitnesses):
        self.solution = solution
        self.fitnesses = fitnesses

    def __hash__(self):
        return hash(tuple(self.solution))

    def dominates(self, other) -> bool:
        return all(f_here < f_there for f_here, f_there in zip(self.fitnesses, other.fitnesses))


    def __repr__(self):
        return f"{self.solution}, {self.fitnesses}"

    def __eq__(self, other):
        return self.solution == other.solution

def sample_from_probabilities(probabilities) -> NCSolution:
    result = set()
    for (index, probability) in enumerate(probabilities):
        if random.random() < probability:
            result.add(index)
    return result


class NCSampler:
    probabilities_of_existing: np.ndarray

    def __init__(self, probabilities_of_existing: np.ndarray):
        self.probabilities_of_existing = probabilities_of_existing
        self.probabilities_of_existing = scale_to_have_sum(self.probabilities_of_existing, wanted_sum=4)

    def sample(self) -> NCSolution:
        return sample_from_probabilities(self.probabilities_of_existing)

    @classmethod
    def from_PRef(cls, pRef: PRef):
        # assumes that pRef is boolean
        probabilities = np.average(pRef.full_solution_matrix, axis=0)
        return cls(probabilities)


def hot_encode_and_multiply(solution: NCSolution, transition_matrix: np.ndarray) -> np.ndarray:
    hot_encoded = np.zeros(transition_matrix.shape[1], dtype=float)
    hot_encoded[list(solution)] = 1
    return hot_encoded.reshape((1, -1)) @ transition_matrix


class NCMutation:
    transition_matrix: np.ndarray
    disappearance_probability: np.ndarray
    n: int

    def __init__(self, transition_probabilities, disappearance_probability):
        self.transition_matrix = transition_probabilities
        self.disappearance_probability = disappearance_probability
        self.n = transition_probabilities.shape[1]

    def mutated(self, solution: NCSolution) -> NCSolution:
        probabilities = hot_encode_and_multiply(solution, self.transition_matrix)
        probabilities = probabilities.ravel()
        probabilities = scale_to_have_sum_and_max(probabilities,
                                                  wanted_sum=len(solution),
                                                  wanted_max=1 - self.disappearance_probability,
                                                  positions=self.n)
        return sample_from_probabilities(probabilities)


class NCCrossover:
    transition_matrix: np.ndarray

    def __init__(self, transition_probabilities):
        self.transition_matrix = transition_probabilities

    def crossed(self, a: NCSolution, b: NCSolution):
        guaranteed = a.intersection(b)
        considering = a ^ b

        considering_probabilities = hot_encode_and_multiply(considering, self.transition_matrix)
        considering_probabilities = considering_probabilities.ravel()
        wanted_quantity_of_ones = ((len(a) + len(b)) / 2) - len(guaranteed)
        considering_probabilities = scale_to_have_sum_and_max(considering_probabilities,
                                                              wanted_sum=wanted_quantity_of_ones,
                                                              wanted_max=1,
                                                              positions=len(considering))
        child_1 = guaranteed.copy()
        child_2 = guaranteed.copy()

        for considered_index in considering:
            if random.random() < considering_probabilities[considered_index]:
                child_1.add(considered_index)
            else:
                child_2.add(considered_index)

        return child_1, child_2


class NSGAIICustom:
    sampling: NCSampler
    mutation: NCMutation
    crossover: NCCrossover
    probability_of_crossover: float

    fitness_functions: list[Callable[[NCSolution], float]]
    pop_size: int
    eval_budget: int
    unique: bool
    tournament_size: int

    def __init__(self,
                 sampling: NCSampler,
                 mutation: NCMutation,
                 crossover: NCCrossover,
                 probability_of_crossover: float,
                 pop_size: int,
                 eval_budget: int,
                 fitness_functions: list[Callable],
                 unique: bool,
                 tournament_size: int):
        self.sampling = sampling
        self.mutation = mutation
        self.crossover = crossover
        self.probability_of_crossover = probability_of_crossover
        self.pop_size = pop_size
        self.eval_budget = eval_budget
        self.fitness_functions = fitness_functions
        self.unique = unique
        self.tournament_size = tournament_size

    def make_unique_population(self, yielder, required_quantity):
        result = set()

        for child in yielder:
            result.add(child)
            if len(result) >= required_quantity:
                return result

    def make_non_unique_population(self, yielder, required_quantity):
        result = list()
        for child in yielder:
            result.append(child)
            if len(result) >= required_quantity:
                return result

    def make_population(self, yielder, required_quantity):
        if self.unique:
            return self.make_unique_population(yielder, required_quantity)
        else:
            return self.make_non_unique_population(yielder, required_quantity)

    def run(self, verbose: bool = False) -> list[EvaluatedNCSolution]:
        def log(msg):
            if verbose:
                print(msg)
        used_evaluations = [0]

        def with_fitnesses(solution: NCSolution) -> EvaluatedNCSolution:
            fitnesses = tuple(f(solution) for f in self.fitness_functions)
            used_evaluations[0] += 1
            return EvaluatedNCSolution(solution, fitnesses)

        def sampler_yielder():
            while True:
                yield with_fitnesses(self.sampling.sample())

        log("Beginning of NC process")
        population = self.make_population(yielder=sampler_yielder(), required_quantity=self.pop_size)

        while (used_evaluations[0] < self.eval_budget):
            population = self.make_next_generation(population, with_fitnesses)
            log(f"Used evals: {used_evaluations[0]}")

        pareto_fronts = self.get_pareto_fronts(population)

        if self.unique:
            return list(set(pareto_fronts[0]))
        return pareto_fronts[0]

    def get_pareto_fronts(self, population: Iterable[EvaluatedNCSolution]) -> list[list[EvaluatedNCSolution]]:
        pareto_fronts = defaultdict(list)

        subs = dict()
        dom_count = dict()

        for p in population:
            subs[p] = list()
            dom_count[p] = 0

            for q in population:
                if p.dominates(q):
                    subs[p].append(q)
                elif q.dominates(p):
                    dom_count[p] += 1

            if dom_count[p] == 0:
                pareto_fronts[0].append(p)

        i = 0
        while True:
            F_i = pareto_fronts[i]
            if len(F_i) < 1:
                break
            Q = list()
            for p in F_i:
                for q in subs[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        Q.append(q)
            pareto_fronts[i+1] = Q
            i += 1

        result_pareto_fronts = [list() for _  in pareto_fronts]
        for front_index, front_members in pareto_fronts.items():
            result_pareto_fronts[front_index] = front_members
        return result_pareto_fronts

    def make_next_generation(self, population: Iterable[EvaluatedNCSolution], evaluator):
        pareto_fronts = self.get_pareto_fronts(population)

        indices_and_ranks = [(index, rank)
                             for rank, front in enumerate(pareto_fronts)
                             for index, _ in enumerate(front)]

        def tournament_select_one() -> EvaluatedNCSolution:
            candidates = random.choices(indices_and_ranks, k=self.tournament_size)
            winner_index, winner_pareto_index = min(candidates, key=utils.second)
            return pareto_fronts[winner_pareto_index][winner_index]

        def make_child_asexually():
            parent = tournament_select_one()
            mutated = self.mutation.mutated(parent.solution)
            return evaluator(mutated)

        def make_child_sexually():
            parent_1, parent_2 = tournament_select_one(), tournament_select_one()
            child_1, child_2 = self.crossover.crossed(parent_1.solution, parent_2.solution)
            child_1 = self.mutation.mutated(child_1)
            child_2 = self.mutation.mutated(child_2)
            return evaluator(child_1), evaluator(child_2)

        def child_yielder():
            while True:
                if random.random() < self.probability_of_crossover:
                    child_1, child_2 = make_child_sexually()
                    yield child_1
                    yield child_2
                else:
                    yield make_child_asexually()

        children = self.make_population(child_yielder(), required_quantity=self.pop_size - len(pareto_fronts[0]))

        if self.unique:
            return children.union(pareto_fronts[0])
        else:
            return children + pareto_fronts[0]  # elitist


def check_dummy():
    problem = RoyalRoad(5)

    # then we make a pRef
    pRef = get_pRef_from_metaheuristic(problem=problem,
                                       sample_size=10000,
                                       which_algorithm="GA",
                                       unique=True,
                                       verbose=True)

    ground_truth_atomicity_evaluator = TraditionalPerturbationLinkage(problem)
    ground_truth_atomicity_evaluator.set_solution(pRef.get_best_solution())

    transition_matrix = ground_truth_atomicity_evaluator.linkage_table

    def simplicity(sol):
        return len(sol)

    def mean_fitness(sol):
        ps_values = np.full(shape=problem.search_space.amount_of_parameters, fill_value=-1, dtype=int)
        ps_values[list(sol)] = 1
        ps = PS(ps_values)
        return -np.average(pRef.fitnesses_of_observations(ps))

    def atomicity(sol):
        if len(sol) < 2:
            return 0
        linkages = [transition_matrix[a, b] for a, b in itertools.combinations(sol, r=2)]
        return -np.average(linkages)


    def metric_a(sol):
        return abs(len(sol) - 10)

    def metric_b(sol):
        return abs(len(sol))

    def metric_c(sol):
        if len(sol) < 2:
            return 1000
        vals = list(sol)
        vals.sort()
        distances = [big-small for big, small in zip(vals[1:], vals[0:])]
        return -np.average(np.square(np.array(distances)))

    algorithm = NSGAIICustom(sampling=NCSampler.from_PRef(pRef),
                             mutation=NCMutation(transition_probabilities=transition_matrix,
                                                 disappearance_probability=0.1),
                             crossover=NCCrossover(transition_probabilities=transition_matrix),
                             probability_of_crossover=0.5,
                             eval_budget=1000,
                             pop_size=100,
                             tournament_size=3,
                             fitness_functions=[metric_a, metric_b],
                             unique=True
                             )

    pss = algorithm.run(verbose=True)
    for ps in pss:
        print(ps.solution, ps.fitnesses)
    return pss


check_dummy()
