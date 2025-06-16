import random
from collections import defaultdict
from typing import Callable, Iterable, Optional, TypeAlias

import numpy as np

import utils

NCSolution: TypeAlias = set[int]


class EvaluatedNCSolution:
    solution: NCSolution
    fitnesses: tuple[float]

    def __init__(self, solution, fitnesses):
        self.solution = solution
        self.fitnesses = fitnesses

    def __hash__(self):
        return hash(self.fitnesses)  # is this a bad idea?

    def dominates(self, other) -> bool:
        return all(f_here <= f_there for f_here, f_there in zip(self.fitnesses, other.fitnesses))

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
    def __init__(self):
        pass

    def sample(self) -> NCSolution:
        raise NotImplementedError()


class NCSamplerSimple(NCSampler):
    probabilities: np.ndarray
    allow_empty: bool

    def __init__(self,
                 probabilities: np.ndarray,
                 allow_empty: bool = False):
        super().__init__()
        self.probabilities = probabilities
        self.allow_empty = allow_empty

    @classmethod
    def with_average_quantity(cls, quantity: float, genome_size: int, allow_empty: bool = False):
        probabilities = np.ones(genome_size) * quantity / genome_size
        return cls(probabilities, allow_empty)

    @classmethod
    def equal_probability(cls, n):
        probabilities = np.ones(n) / n
        return cls(probabilities)

    def sample(self) -> NCSolution:
        produced = sample_from_probabilities(self.probabilities)

        # we don't want empty solutions
        if len(produced) == 0 and not self.allow_empty:
            produced.add(random.choice(range(len(self.probabilities))))
        return produced


class NCMutation:
    def __init__(self):
        pass

    def mutated(self, solution: NCSolution) -> NCSolution:
        raise NotImplementedError()


class NCMutationSimple(NCMutation):
    n: int
    mutation_rate: float

    def __init__(self, n: int, mutation_rate: Optional[float] = None):
        super().__init__()
        self.n = n
        self.mutation_rate = mutation_rate if mutation_rate is not None else 1 / n

    def mutated(self, solution: NCSolution) -> NCSolution:
        result = solution.copy()
        for index in range(self.n):
            toggle = random.random() < self.mutation_rate
            if not toggle:
                continue
            if index in result:
                result.remove(index)
            else:
                result.add(index)
        return result


class NCCrossover:

    def __init__(self):
        pass

    def crossed(self, a: NCSolution, b: NCSolution) -> (NCSolution, NCSolution):
        raise NotImplementedError()


class NCCrossoverSimple(NCCrossover):
    swap_probability: float

    def __init__(self, swap_probability: float):
        super().__init__()
        self.swap_probability = swap_probability

    def crossed(self, a: NCSolution, b: NCSolution):
        guaranteed = a.intersection(b)

        child_1 = guaranteed.copy()
        child_2 = guaranteed.copy()

        def add_to_children(main_parent, preferred_child, other_child):
            for exclusive in main_parent - guaranteed:
                if random.random() < self.swap_probability:
                    other_child.add(exclusive)
                else:
                    preferred_child.add(exclusive)

        add_to_children(a, child_1, child_2)
        add_to_children(b, child_2, child_1)
        return child_1, child_2


def make_non_unique_population(yielder, required_quantity):
    result = list()
    for child in yielder:
        result.append(child)
        if len(result) >= required_quantity:
            return result


class NSGAIICustom:
    sampling: NCSampler
    mutation: NCMutation
    crossover: NCCrossover
    probability_of_crossover: float

    mo_fitness_function: Callable[[NCSolution], tuple[float]]
    pop_size: int
    eval_budget: int
    unique: bool
    tournament_size: int
    culler: Callable
    verbose: bool

    def __init__(self,
                 sampling: NCSampler,
                 mutation: NCMutation,
                 crossover: NCCrossover,
                 probability_of_crossover: float,
                 pop_size: int,
                 eval_budget: int,
                 mo_fitness_function: Callable[[NCSolution], tuple[float]],
                 unique: bool,
                 tournament_size: int,
                 culler: Optional[
                     Callable[[Iterable[EvaluatedNCSolution], int], Iterable[EvaluatedNCSolution]]] = None,
                 verbose: bool = False):
        self.sampling = sampling
        self.mutation = mutation
        self.crossover = crossover
        self.probability_of_crossover = probability_of_crossover
        self.pop_size = pop_size
        self.eval_budget = eval_budget
        self.mo_fitness_function = mo_fitness_function
        self.unique = unique
        self.tournament_size = tournament_size
        self.culler = culler if culler is not None else self.default_culler
        self.verbose = verbose

    def log(self, msg: str):
        if self.verbose:
            print("NSGAIICustom -> " + msg)

    def default_culler(self, population: Iterable[EvaluatedNCSolution], quantity_required: int) -> Iterable[
        EvaluatedNCSolution]:
        return list(population)[:quantity_required]

    def make_unique_population(self, yielder, required_quantity):
        result = set()

        for child in yielder:
            result.add(child)
            if len(result) >= required_quantity:
                return result

    def make_population(self, yielder, required_quantity):
        if self.unique:
            return self.make_unique_population(yielder, required_quantity)
        else:
            return make_non_unique_population(yielder, required_quantity)

    def run(self) -> list[EvaluatedNCSolution]:

        used_evaluations = [0]

        def with_fitnesses(solution: NCSolution) -> EvaluatedNCSolution:
            fitnesses = self.mo_fitness_function(solution)
            used_evaluations[0] += 1
            return EvaluatedNCSolution(solution, fitnesses)

        def sampler_yielder():
            while True:
                yield with_fitnesses(self.sampling.sample())

        self.log("Beginning of NC process")
        population = self.make_population(yielder=sampler_yielder(), required_quantity=self.pop_size)

        while (used_evaluations[0] < self.eval_budget):
            population = self.make_next_generation(population, with_fitnesses)
            self.log(f"Used evals: {used_evaluations[0]}")
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
            pareto_fronts[i + 1] = Q
            i += 1

        pareto_front_lists = [list() for _ in pareto_fronts]
        for front_index, front_members in pareto_fronts.items():
            pareto_front_lists[front_index] = front_members

        return pareto_front_lists

    def child_yielder(self,
                      population,
                      pareto_fronts,
                      evaluator):
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

        while True:
            if random.random() < self.probability_of_crossover:
                child_1, child_2 = make_child_sexually()
                yield child_1
                yield child_2
            else:
                yield make_child_asexually()

    def add_to_population(self, pop, new_elements):
        if self.unique:
            pop.update(new_elements)
        else:
            pop.extend(new_elements)

    def union_of_populations(self, pop, new_elements: Iterable[EvaluatedNCSolution]):
        if self.unique:
            return pop.union(new_elements)
        else:
            new_pop = pop.copy()
            new_pop.extend(new_elements)
            return new_pop

    def make_next_generation(self, population: Iterable[EvaluatedNCSolution], evaluator):
        pareto_fronts = self.get_pareto_fronts(population)

        Q = self.make_population(self.child_yielder(population, pareto_fronts, evaluator),
                                 required_quantity=self.pop_size)

        R = self.union_of_populations(population, Q)
        fronts_of_R = self.get_pareto_fronts(R)  # I don't like that there's 2 constructions of pareto fronts!

        # then truncation selection by front
        final_population = set() if self.unique else list()
        front_that_was_excluded = None

        for front_index, front in enumerate(fronts_of_R):
            if len(final_population) + len(front) < self.pop_size:
                self.add_to_population(final_population, front)
            else:
                front_that_was_excluded = front

        if front_that_was_excluded is not None:
            self.add_to_population(final_population,
                                   self.culler(front_that_was_excluded,
                                               quantity_required=self.pop_size - len(final_population)))
        return final_population


def check_dummy():
    def sum_of_values(sol):
        return -sum(sol)

    def simplicity(sol):
        return float(len(sol))

    def distances_between_values(sol) -> float:
        if len(sol) < 2:
            return 1000
        vals = list(sol)
        vals.sort()
        distances = [big - small for big, small in zip(vals[1:], vals[0:])]
        return -np.average(np.square(np.array(distances)))

    n = 16

    def quantity_of_divisors(sol):
        def find_divisors(num):
            return {d for d in range(2, num + 1) if num % d == 0}

        if sol:
            all_divisors = set.union(*(find_divisors(num) for num in sol))
            return len(all_divisors)
        else:
            return 0

    def get_metrics(ps: NCSolution) -> tuple[float]:
        return (sum_of_values(ps), quantity_of_divisors(ps))

    algorithm = NSGAIICustom(sampling=NCSamplerSimple.with_average_quantity(n / 2, genome_size=n),
                             mutation=NCMutationSimple(n),
                             crossover=NCCrossoverSimple(swap_probability=1 / n),
                             probability_of_crossover=0.5,
                             eval_budget=5000,
                             pop_size=100,
                             tournament_size=3,
                             mo_fitness_function=get_metrics,
                             unique=True
                             )

    pss = algorithm.run(verbose=True)
    pss = list(set(pss))
    pss.sort(key=lambda x: x.fitnesses[0])
    for ps in pss:
        print(ps.solution, ps.fitnesses)
    return pss
