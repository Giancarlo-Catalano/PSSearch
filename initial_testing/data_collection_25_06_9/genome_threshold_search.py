import heapq
import itertools
from dataclasses import dataclass
from typing import Optional, Iterable, Any

import numpy as np
from scipy.stats import wasserstein_distance, mannwhitneyu

from Core.PRef import PRef
from Core.PSMetric.Linkage.ValueSpecificMutualInformation import GlobalLinkageBasedOnMutualInformation
from Gian_experimental.NSGAIICustom.CustomOperators import NCSamplerFromPRef, NCMutationCounterproductive, \
    NCCrossoverTransition
from Gian_experimental.NSGAIICustom.NSGAIICustom import EvaluatedNCSolution, NCSamplerSimple, NCMutationSimple, \
    NCCrossoverSimple, NCSolution, NSGAIICustom
from Gian_experimental.NSGAIICustom.evolving_genome_threshold import SampleWithFixedGT, SimpleSampleGT, MutateExceptGT, SimpleMutateGT, SimpleCrossoverGT
from Gian_experimental.NSGAIICustom.testing_in_vitro.SPRef import SPRef, OptimisedSPref
from PolishSystem.OperatorsBasedOnSimilarities.similarities_utils import gian_get_similarities, get_transition_matrix


@dataclass
class PolishSearchSettings:
    code_name: str
    population_size: int
    evaluation_budget: int

    cluster_info_file_name: str

    use_custom_sampling_operator: bool = False
    use_custom_mutation_operator: bool = False
    use_custom_crossover_operator: bool = False

    use_custom_atomicity: bool = False
    genome_threshold: Any = None  # None, int, or "auto"

    include_ps_len: bool = False
    include_sample_quantity: bool = False
    include_wasserstein_distance: bool = False
    include_diff_median: bool = False
    include_atomicity: bool = False

    include_p_value: bool = False
    include_mean_fitness: bool = False
    include_variance: bool = False



    def to_dict(self):
        return vars(self)


    def get_quantity_of_objectives(self):
        return sum([self.include_ps_len,
                    self.include_atomicity,
                    self.include_p_value,
                    self.include_sample_quantity,
                    self.include_mean_fitness,
                    self.include_diff_median,
                    self.include_wasserstein_distance,
                    self.include_variance])

    def auto_fill_code(self):

        def keywords_in_brackets(keywords):
            return "["+ " ".join(keywords) + "]"

        # operators
        operator_keywords = []
        operator_keywords.append("CS" if self.use_custom_sampling_operator else "OS")
        operator_keywords.append("CM" if self.use_custom_mutation_operator else "OM")
        operator_keywords.append("CC" if self.use_custom_crossover_operator else "OC")

        # metrics keywords
        metrics_keywords = []
        if self.include_ps_len:
            metrics_keywords.append("Len")

        if self.include_sample_quantity:
            metrics_keywords.append("SSize")

        if self.include_mean_fitness:
            metrics_keywords.append("MFit")

        if self.include_p_value:
            metrics_keywords.append("PVal")

        if self.include_wasserstein_distance:
            metrics_keywords.append("WDist")

        if self.include_diff_median:
            metrics_keywords.append("DMed")

        if self.include_variance:
            metrics_keywords.append("WVar")

        if self.include_atomicity:
            if self.use_custom_atomicity:
                metrics_keywords.append("CAtom")
            else:
                metrics_keywords.append("OldAtom")

        # Genome threshold
        genome_keywords = []
        genome_keywords.append("G"+str(self.genome_threshold))

        self.code_name = "".join([keywords_in_brackets(operator_keywords), keywords_in_brackets(metrics_keywords), keywords_in_brackets(genome_keywords)])


class HashedSolution:
    solution: NCSolution

    def __init__(self,
                 sol):
        self.solution = sol

    def __hash__(self):
        return sum(self.solution) % 7787

    def __eq__(self, other):
        return self.solution == other.solution


def make_metrics_cached(metrics):
    cached_values = dict()

    def get_values(ps):
        wrapped = HashedSolution(ps)
        if wrapped in cached_values:
            return cached_values[wrapped]
        else:

            value = metrics(ps)
            cached_values[wrapped] = value
            return value

    return get_values


def search_for_pss_using_genome_threshold(train_session_data: PRef,
                                          optimised_session_data: OptimisedSPref,
                                          search_settings: PolishSearchSettings,
                                          old_atomicity_metric: Optional[GlobalLinkageBasedOnMutualInformation]) -> list[EvaluatedNCSolution]:

    n = train_session_data.search_space.amount_of_parameters
    def tie_breaker(population: Iterable[EvaluatedNCSolution], quantity_required: int):
        return heapq.nsmallest(iterable=population, key=lambda x: x.fitnesses[0], n=quantity_required)

    similarities = gian_get_similarities(search_settings.cluster_info_file_name)
    transition_matrix = get_transition_matrix(similarities)

    custom_sampling = NCSamplerFromPRef.from_PRef(train_session_data)
    custom_mutation = NCMutationCounterproductive(transition_matrix)
    custom_crossover = NCCrossoverTransition(transition_matrix)


    atomicity = None
    if search_settings.use_custom_atomicity:
        def atomicity_based_on_similarity(ps):
            if len(ps) < 2:
                return -1000
            else:
                linkages = [similarities[a, b] for a, b in itertools.combinations(ps, r=2)]
                return np.average(linkages)
        atomicity = atomicity_based_on_similarity
    else:
        def old_atomicity(ps):
            return old_atomicity_metric.get_atomicity_of_set(ps)

        atomicity = old_atomicity


    traditional_sampling = NCSamplerSimple.with_average_quantity(3, genome_size=n, allow_empty=False)
    traditional_mutation = NCMutationSimple(n)
    traditional_crossover = NCCrossoverSimple(swap_probability=1 / n)

    quantity_of_objectives = search_settings.get_quantity_of_objectives()

    def get_metrics(ps: NCSolution) -> tuple[float]:
        if ps.genome_threshold == "auto":
            raise NotImplementedError()
        matching, non_matching = optimised_session_data.partition(ps, threshold=ps.genome_threshold)
        len_m, len_nm = len(matching), len(non_matching)

        if len(matching) < 1000:  # a constraint
            return tuple([1000.0]*quantity_of_objectives)

        result = []

        if search_settings.include_ps_len:
            result.append(len(ps))

        if search_settings.include_sample_quantity:
            result.append(-len_m)

        if search_settings.include_mean_fitness:
            result.append(-np.average(matching))

        if search_settings.include_p_value:
            test = mannwhitneyu(matching, non_matching, alternative="greater")
            result.append(test.pvalue)

        if search_settings.include_wasserstein_distance:
            distance = wasserstein_distance(matching, non_matching) if min(len(matching), len(non_matching)) > 2 else 0
            result.append(-distance)

        if search_settings.include_diff_median:
            median_diff = np.median(matching) - np.median(non_matching)
            result.append(-median_diff)

        if search_settings.include_variance:
            weighted_variance_score = (np.var(matching) * len_m + np.var(non_matching) * len_nm) / (len_m + len_nm)
            result.append(weighted_variance_score)

        if search_settings.include_atomicity:
            atomicity_score = atomicity(ps)
            result.append(-atomicity_score)

        return tuple(result)

    sampling_on_set = custom_sampling if search_settings.use_custom_sampling_operator else traditional_sampling
    mutation_on_set = custom_mutation if search_settings.use_custom_mutation_operator else traditional_mutation
    crossover_on_set = custom_crossover if search_settings.use_custom_mutation_operator else traditional_crossover



    if search_settings.genome_threshold == "auto":
        sampling_to_use = SimpleSampleGT(original_sampler=sampling_on_set)
        mutation_to_use = SimpleMutateGT(original_mutation=mutation_on_set)
        crossover_to_use = SimpleCrossoverGT(original_crossover=crossover_on_set)
    else:
        # if we don't want to evolve the genome threshold, just initialise them all at the same value
        # then don't allow mutation to change it
        # crossover cannot produce anything new
        sampling_to_use = SampleWithFixedGT(original_sampler=sampling_on_set, genome_threshold=search_settings.genome_threshold)
        mutation_to_use = MutateExceptGT(original_mutation=mutation_on_set)
        crossover_to_use = SimpleCrossoverGT(original_crossover=crossover_on_set)


    algorithm = NSGAIICustom(sampling=sampling_to_use,
                             mutation=mutation_to_use,
                             crossover=crossover_to_use,
                             probability_of_crossover=0.5,
                             eval_budget=search_settings.evaluation_budget,
                             pop_size=search_settings.population_size,
                             tournament_size=3,
                             mo_fitness_function=make_metrics_cached(get_metrics),
                             unique=True,
                             verbose=False,
                             culler=tie_breaker)


    run_results = algorithm.run()
    return run_results



def results_to_json(results_of_run: Iterable[EvaluatedNCSolution],
                    test_session_data: OptimisedSPref,
                    search_settings: PolishSearchSettings) -> list[dict]:
    def ps_to_json(run_result: EvaluatedNCSolution) -> dict:
        ps = run_result.solution
        matching, non_matching = test_session_data.partition(ps, threshold=ps.genome_threshold)
        can_do_stats = min(len(matching), len(non_matching)) > 2

        serialisable_ps = sorted(list(ps))
        if not can_do_stats:
            return {"ps": serialisable_ps,
                    "samples": len(matching) / len(non_matching),
                    "genome_threshold": None if ps.genome_threshold is None else int(ps.genome_threshold)}
        else:
            test = mannwhitneyu(matching, non_matching, alternative="greater")
            return {"ps": serialisable_ps,
                    "genome_threshold": None if ps.genome_threshold is None else int(ps.genome_threshold),
                    "median_diff": np.median(matching) - np.median(non_matching),
                    "p_value": test.pvalue,
                    "samples": len(matching) / (len(non_matching)+len(matching))}

    return list(map(ps_to_json, results_of_run))