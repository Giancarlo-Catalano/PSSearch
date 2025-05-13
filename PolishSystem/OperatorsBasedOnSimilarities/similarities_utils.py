import numpy as np
from collections import Counter
import gzip, pickle
from scipy.stats.qmc import MultivariateNormalQMC
from sklearn.cluster import KMeans

from Core.PS import PS, STAR


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name):
        # remap deprecated or changed module names
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def gian_load_all_pickled_objects(file_path):
    objects = []
    with gzip.open(file_path, "rb") as f:
        unpickler = CustomUnpickler(f)
        while True:
            try:
                obj = unpickler.load()
                objects.append(obj)
            except EOFError:
                break
    return objects

def gian_get_similarities(cluster_info_file):
    contents_of_pickle = gian_load_all_pickled_objects(cluster_info_file)
    which_cluster, centers_embeddings, similarities, _, _, _ = contents_of_pickle

    return similarities

def normalise_vector(vector):
    m = np.min(vector)
    M = np.max(vector)
    return (vector-m) / (M-m)
def normalise_every_row(table):
    return np.array([normalise_vector(row) for row in table])


def get_transition_matrix(similarities):
    def normalise_vector(vector):
        m = np.min(vector)
        M = np.max(vector)
        return (vector - m) / (M - m)

    def normalise_every_row(table):
        return np.array([normalise_vector(row) for row in table])

    transition_matrix = normalise_every_row(similarities)

    transition_matrix = np.array([row / np.sum(row) for row in transition_matrix])
    return transition_matrix

def scale_to_have_sum(vec: np.ndarray, wanted_sum: float):
    return wanted_sum * vec / np.sum(vec)

def scale_to_have_sum_and_max(vec: np.ndarray, wanted_sum, wanted_max, positions: int):
    with_sum_one = vec / np.sum(vec)
    current_max = np.max(with_sum_one)
    # current sum = 1
    n = positions

    a_num = wanted_max * n - wanted_sum
    b_num = current_max * wanted_sum - wanted_max
    denom = current_max * n - 1

    result = (with_sum_one * a_num + b_num) / denom

    return result




def sample_PS_from_probabilties_for_local(probabilities):
    return np.random.random(len(probabilities)) < probabilities

def sample_PS_from_probabilties_for_global(probabilities) -> np.ndarray:
    return np.random.random(len(probabilities)) < probabilities


def from_global_to_zeroone(glob):
    return np.array(glob!=STAR, dtype=int)
