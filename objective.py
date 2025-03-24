"""
DRAFT: needs adapting for boolean encoding and matrix calculations, refactoring, tests and review
"""
import numpy as np

import os
from pathlib import Path

current_dir = os.getcwd()

os.chdir(Path(__file__).parent)
from utils import find_matching

os.chdir(current_dir)

def mean_fitness(pattern, sessions_train, scores, stype=0):
    """
    find mean fitness of sequences matching a pattern
    :param pattern: no_sequences x dim (-1 for *)
    :param sessions_train: no_patterns x dim (binary)
    :param scores: no_patterns x max_stype
    :param stype: int
    :return: no_patterns (float)
    """
    matching = find_matching(pattern, sessions_train)
    scores_  = scores[:, stype]
    return np.nan_to_num(np.array([scores_[vec].mean() for vec in matching.T]), 0)
    # return np.nan_to_num(np.apply_along_axis(lambda vec: scores[vec, stype].mean(), 0, find_matching(pattern, sessions_train)), 0)

def simplicity(pattern):
    return (~pattern).mean(-1)


def simplicity_positives_scaled(pattern, sessions_train):
    """
    Simplicity is 1 - complexity, where complexity is the ratio of positive coordinates to the maximum number of positives matching a pattern
    :param pattern:
    :return:
    """
    return 1 - pattern.sum() / np.apply_along_axis(lambda vec: sessions_train[vec].sum(-1).max(), 0, find_matching(pattern, sessions_train))

# def simplicity_weighted(pattern, w=1, max_w=None):
#     """
#     Penalize fixed elements propotionally to the ratio of their occurence in the data
#     Simplicity is 1 - complexity, where complexity is the number of ones + reweighted with w number of zeros
#     :param pattern:
#     :param w:
#     :return:
#     """
#     if max_w is None:
#         max_w = pattern.shape[1]
#     return max_w - (pattern + (~pattern == 0) * w).sum(-1)


def isolate(pattern, k):
    res = np.zeros(pattern.shape).astype(bool)
    res[:, k] = pattern[:, k]
    return res

def exclude(pattern, k):
    res = pattern.copy()
    res[:, k] = np.False_
    return res

def benefit(pattern, sessions_train, scores, norm=0):
    if norm == 0:
        min_ = scores[:, 0].min()
        max_ = scores[:, 0].max()
        range_ = max_ - min_
        scores_norm = np.nan_to_num(np.apply_along_axis(
            lambda vec: ((scores[vec, 0] - min_) / range_).sum(), 0, find_matching(pattern, sessions_train)), 0)
    if norm > 0:
        scores_norm = np.nan_to_num(np.apply_along_axis(
            lambda vec: scores[vec].sum(), 0, find_matching(pattern, sessions_train)), 0)
    return scores_norm

def contribution(pattern, sessions_train, scores, k, norm=0):
    return benefit(pattern, sessions_train, scores, norm) * np.log(benefit(pattern, sessions_train, scores, norm) / benefit(isolate(pattern, k), sessions_train, scores, norm) / benefit(exclude(pattern, k), sessions_train, scores, norm))


def atomicity_(pattern, sessions_train, scores, norm=0): # TODO
    N = pattern.shape[1]
    return np.vstack([contribution(pattern, sessions_train, scores, k, norm) for k in range(N)])

def atomicity(pattern, sessions_train, scores, norm=0):
    a = atomicity_(pattern, sessions_train, scores, norm).T
    return np.where(pattern, a, np.inf).min(1)

def atom_sim(pattern, embedd_centers):  # TODO
    selected = embedd_centers[pattern[0] >= 0, :]
    cov = selected @ selected.T
    var = np.sqrt(np.diag(cov))
    var[var <= 0] = 1e-10
    corr = cov / var.reshape(-1, 1) / var.reshape(1, -1)
    return corr.min()

def atom_sim_pos(pattern, embedd_centers):  # TODO
    selected = embedd_centers[pattern[0] > 0, :]
    cov = selected @ selected.T
    var = np.sqrt(np.diag(cov))
    var[var <= 0] = 1e-10
    corr = cov / var.reshape(-1, 1) / var.reshape(1, -1)
    return corr.min()


def objective(pattern, sessions_train, scores, fitness_type, simplicity_type, atomicity_type, s_w=1, a_w=1, sim_kwargs=dict(), atom_kwargs=dict()):
    # TODO
    """

    :param pattern:
    :param fitness_type: 0 (sigmoid), 1 (softmax), 2 (dot-product)
    :param simplicity_type: simplicity, simplicity_positives_only, simplicity_positives_scaled, simplicity_weighted
    :param atomicity_type: atomicity, atom_sim, atom_sim_pos
    :param s_w: weight
    :param a_w: weight
    :param simplicity_kwargs:
    :return:
    """
    return mean_fitness(pattern, sessions_train, scores, fitness_type) + s_w * eval(simplicity_type)(pattern, **sim_kwargs) + a_w * eval(
        atomicity_type, **atom_kwargs)(pattern)

# if __name__ == '__main__':
#     # TODO
#     import gzip, pickle
#     from pathlib import Path
#
#     with gzip.open(f'{Path(__file__).parent}/LightGCN/data/amazon-book/results.pcklz', 'rb') as f:
#         sequences = pickle.load(f)
#         targets = pickle.load(f)
#         recommendations = pickle.load(f)
#         embeddings = pickle.load(f)
#         target_scores = pickle.load(f)
#         target_scores1 = pickle.load(f)
#         target_scores2 = pickle.load(f)
#
#     N_selected = 100
#     method = 'qmc'
#
#     with open(f'{Path(__file__).parent}/LightGCN/data/amazon-book/cluster_IDs_{N_selected}_{method}.pkl', 'rb') as f:
#         kmeans_clusters = pickle.load(f)
#     sessions_train = np.genfromtxt(f'{Path(__file__).parent}/LightGCN/data/amazon-book/many_hot_vectors_{N_selected}_{method}.csv',
#                                    delimiter=',').astype(int)
#     scores = np.genfromtxt(f'{Path(__file__).parent}/LightGCN/data/amazon-book/fitness_{N_selected}_{method}.csv', delimiter=',')
#
#     example_pattern = sessions_train[np.random.choice(sessions_train.shape[0], size=1)]
#     example_pattern[:, np.random.choice([True, False], size=example_pattern.shape[1])] = -1
#
#     max_w = N_selected - sessions_train.sum(1).max() * (1 - sessions_train.mean())
#     embedd_centers = np.nan_to_num(
#         np.vstack([embeddings[kmeans_clusters == k].mean(0).reshape(1, -1) for k in range(N_selected)]), 0)
#
#     def unit_test_print(command):
#         print(command)
#         print(eval(command))
#         print("\n\n")
#
#     unit_test_print("find_matching(example_pattern)")
#     unit_test_print("mean_fitness(example_pattern, 0)")
#     unit_test_print("mean_fitness(example_pattern, 1)")
#     unit_test_print("mean_fitness(example_pattern, 2)")
#     unit_test_print("simplicity(example_pattern)")
#     unit_test_print("simplicity_positives_only(example_pattern)")
#     unit_test_print("simplicity_positives_scaled(example_pattern)")
#
#     unit_test_print("max_w")
#     unit_test_print("simplicity_weighted(example_pattern)")
#     unit_test_print("exclude(example_pattern, 0)")
#     unit_test_print("isolate(example_pattern, 0)")
#     unit_test_print("benefit(example_pattern, 0)")
#     unit_test_print("benefit(example_pattern, 1)")
#     unit_test_print("benefit(example_pattern, 2)")
#     unit_test_print("atomicity(example_pattern, 0)")
#     unit_test_print("atomicity(example_pattern, 1)")
#     unit_test_print("atomicity(example_pattern, 2)")
#     unit_test_print("atom_sim(example_pattern)")
#     unit_test_print("atom_sim_pos(example_pattern)")
#     unit_test_print('objective(example_pattern, 0, "simplicity_positives_only", "atom_sim_pos")')

