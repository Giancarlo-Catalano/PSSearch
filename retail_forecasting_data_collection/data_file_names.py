import os
import sys

script_location = sys.argv[0]
script_folder = os.path.dirname(script_location)

def convert_path_from_relative_to_absolute(relative_to_main: str) -> str:
    return os.path.join(script_folder, relative_to_main)


size_of_vectors = 250
clustering_method = "random"

data_path = os.path.join(convert_path_from_relative_to_absolute("retail_forecasting_data_collection"), "data")
fitness_values_path = os.path.join(data_path, f"fitness_{size_of_vectors}_{clustering_method}.csv")
vector_path = os.path.join(data_path, f"many_hot_vectors_{size_of_vectors}_{clustering_method}.csv")
cluster_info_path = os.path.join(data_path, f"cluster_info_{size_of_vectors}_{clustering_method}.pkl")