import json
import os

import utils
from Gian_experimental.NSGAIICustom.testing_in_vitro.SPRef import OptimisedSPref
from PolishSystem.read_data import get_pRef_from_vectors
from initial_testing.data_collection_25_06_9.genome_threshold_search import PolishSearchSettings, \
    search_for_pss_using_genome_threshold, results_to_json

cluster_info_file_name = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting\cluster_info_250_random.pkl"
data_path = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting"
results_path = r"C:\Users\gac8\PycharmProjects\PSSearch\initial_testing\data_collection_25_06_9\results\v2"

def run(destination_path):
    print("Starting data collection")

    all_new = PolishSearchSettings(code_name="all_new",
                                   population_size=100,
                                   evaluation_budget=5000,
                                   genome_threshold=3,
                                   cluster_info_file_name=cluster_info_file_name,
                                   use_custom_atomicity=True,
                                   use_custom_sampling_operator=True,
                                   use_custom_mutation_operator=True,
                                   use_custom_crossover_operator=True,
                                   include_ps_len=True,
                                   include_sample_quantity=True,
                                   include_wasserstein_distance=True,
                                   include_diff_median=True,
                                   include_atomicity=True)

    all_old = PolishSearchSettings(code_name="all_old",
                                   population_size=100,
                                   evaluation_budget=5000,
                                   cluster_info_file_name=cluster_info_file_name,
                                   include_sample_quantity=True,
                                   include_variance=True,
                                   include_atomicity=True)


    configs = [all_old, all_new]

    def in_path(path):
        return os.path.join(data_path, path)
    train_pRef = get_pRef_from_vectors(name_of_vectors_file=in_path("train_many_hot_vectors_250_random.csv"),
                                       name_of_fitness_file=in_path("train_fitness_250_random.csv"),
                                       column_in_fitness_file=2)
    test_pRef = get_pRef_from_vectors(name_of_vectors_file=in_path("test_many_hot_vectors_250_random.csv"),
                                      name_of_fitness_file=in_path("test_fitness_250_random.csv"),
                                      column_in_fitness_file=2)

    train_SPRef = OptimisedSPref.from_pRef(train_pRef)
    test_SPRef = OptimisedSPref.from_pRef(test_pRef)

    collected_data = []
    for config in configs:
        print(f"Running for settings {config.code_name}")
        settings_json = config.to_dict()
        result = search_for_pss_using_genome_threshold(train_session_data=train_pRef,
                                                       optimised_session_data=train_SPRef,
                                                       search_settings=config)
        result_json = results_to_json(result, search_settings=config,
                                      test_session_data=test_SPRef)
        collected_data.append({"config": settings_json,
                "results": result_json})


    destination_file = os.path.join(destination_path, "run_"+utils.get_formatted_timestamp()+".json")
    with utils.open_and_make_directories(destination_file) as file:
        json.dump(collected_data, file, indent=4)

    print("All done!")

for i in range(12):
    print(f"iteration {i}")
    run(results_path)