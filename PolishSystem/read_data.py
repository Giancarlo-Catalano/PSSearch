import gzip
import os
import pickle

#%%
from Core.SearchSpace import SearchSpace
import numpy as np
from Core.PRef import PRef


def get_pRef_from_vectors(name_of_vectors_file: str, name_of_fitness_file: str, column_in_fitness_file: int) -> PRef:
    full_solution_matrix = np.loadtxt(name_of_vectors_file, delimiter=",", dtype=int)
    fitness_array = np.genfromtxt(name_of_fitness_file, delimiter=",", dtype=float, usecols=column_in_fitness_file)
    search_space = SearchSpace(2 for _ in range(full_solution_matrix.shape[1]))
    return PRef(full_solution_matrix=full_solution_matrix,
                fitness_array=fitness_array,
                search_space=search_space)

def get_vectors_file_name(data_folder: str, vector_size: int, clustering_method: str) -> str:
    return os.path.join(data_folder, f"many_hot_vectors_{vector_size}_{clustering_method}.csv")

def get_fitness_file_name(data_folder: str, vector_size: int, clustering_method: str) -> str:
    return os.path.join(data_folder, f"fitness_{vector_size}_{clustering_method}.csv")


def get_cluster_info_file_name(data_folder: str, vector_size: int, clustering_method: str) -> str:
    return os.path.join(data_folder, f"cluster_info_{vector_size}_{clustering_method}.pkl")


def example_usage_for_read_data():
    folder = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting"
    size = 20
    method = "kmeans"
    fitness_column_to_use = 0

    pRef = get_pRef_from_vectors(name_of_vectors_file=get_vectors_file_name(folder, size, method),
                                 name_of_fitness_file=get_fitness_file_name(folder, size, method),
                                 column_in_fitness_file=fitness_column_to_use)
    best_solution = pRef.get_best_solution()

    print(pRef)


def read_cluster_info_file(cluster_info_file_name: str, user: str):
    if user == "klaudia":
        try:
            with gzip.open(cluster_info_file_name, "rb") as f:
                # fix_imports=True sometimes helps with module remapping
                data = pickle.load(f, fix_imports=True)
            print("Success with gzip using standard pickle.load.")
            return data
        except Exception as e:
            print("load_with_gzip_standard failed:", e)
            return None


    if user == "gian":
        try:
            with gzip.open(cluster_info_file_name, "rb") as f:
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module:str, name):
                        # remap old numpy module name to current
                        print(f"Attempted to find the module {module}")
                        replaced_string = module.replace("numpy._core", "numpy.core")
                        return super().find_class(replaced_string, name)

                data = CustomUnpickler(f).load()
            print("Success with gzip using custom unpickler.")

            return data
        except Exception as e:
            print("load_with_gzip_custom failed:", e)
            return None


    raise NotImplemented


