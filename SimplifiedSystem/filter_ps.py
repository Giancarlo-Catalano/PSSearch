import numpy as np

import utils
from Core.PS import PS, STAR
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage


def keep_biggest(pss: list[PS]) -> [PS]:
    """returns a singleton list containing the pss with the most variables being fixed, (i know it's counterintuitive"""
    """assumes simplicity is the first metric"""
    return utils.top_with_safe_ties(pss, key=lambda x: np.sum(x.values != STAR), lowest=False)


def keep_with_lowest_dependence(pss: list[PS], local_linkage_metric: TraditionalPerturbationLinkage) -> [PS]:
    return utils.top_with_safe_ties(pss, key=lambda x: local_linkage_metric.get_dependence(x), lowest=True)


def keep_with_best_atomicity(pss: list[PS]) -> [PS]:
    return utils.top_with_safe_ties(pss, key=lambda x: x.metric_scores[2])


def keep_middle(pss: list[PS]) -> [PS]:
    # assuming that they are ordered in a sensible way?
    qty_metrics = len(pss[0].metric_scores)
    metrics = [lambda x: x.metric_scores[i] for i in range(1, qty_metrics)]  #
    sorted_pss = utils.sort_by_combination_of(pss, key_functions=metrics)
    middle_index = len(pss) // 2
    return [sorted_pss[middle_index]]


def merge_pss_into_one(pss: list[PS]) -> PS:
    # assumes that no PSS are in disagreement
    pss_matrix = np.array([ps.values for ps in pss])
    final_values = np.max(pss_matrix, axis=0)  # since stars are -1
    return PS(final_values)
