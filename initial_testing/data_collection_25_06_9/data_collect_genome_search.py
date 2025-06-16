import copy
import json
import os
import asyncio
import random

import utils
from Core.PSMetric.Linkage.ValueSpecificMutualInformation import GlobalLinkageBasedOnMutualInformation
from Gian_experimental.NSGAIICustom.testing_in_vitro.SPRef import OptimisedSPref
from PolishSystem.read_data import get_pRef_from_vectors
from initial_testing.data_collection_25_06_9.genome_threshold_search import PolishSearchSettings, \
    search_for_pss_using_genome_threshold, results_to_json

cluster_info_file_name = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting\cluster_info_250_random.pkl"
data_path = r"C:\Users\gac8\PycharmProjects\PSSearch\data\retail_forecasting"
destination_path = r"C:\Users\gac8\PycharmProjects\PSSearch\initial_testing\data_collection_25_06_9\results\v5"


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


def run():
    print("Starting data collection")
    evaluation_budget = 10000

    # baseline = PolishSearchSettings(code_name="baseline",
    #                                 population_size=100,
    #                                 evaluation_budget=evaluation_budget,
    #                                 cluster_info_file_name=cluster_info_file_name,
    #                                 include_ps_len=True,
    #                                 include_mean_fitness=True,
    #                                 include_atomicity=True,
    #                                 use_custom_atomicity=False,
    #                                 genome_threshold=None,
    #                                 )
    # baseline.auto_fill_code()
    #
    # with_new_atomicity = copy.deepcopy(baseline)
    # with_new_atomicity.use_custom_atomicity = True
    # with_new_atomicity.auto_fill_code()
    #
    # with_ss = copy.deepcopy(with_new_atomicity)
    # with_ss.include_sample_quantity = True
    # with_ss.auto_fill_code()
    #
    # with_genome_3 = copy.deepcopy(with_new_atomicity)
    # with_genome_3.genome_threshold = 3
    # with_genome_3.use_custom_sampling_operator = True
    # with_genome_3.use_custom_mutation_operator = True
    # with_genome_3.use_custom_crossover_operator = True
    # with_genome_3.auto_fill_code()
    #
    # with_genome_4 = copy.deepcopy(with_genome_3)
    # with_genome_4.genome_threshold = 4
    # with_genome_4.auto_fill_code()
    #
    # with_genome_5 = copy.deepcopy(with_genome_3)
    # with_genome_5.genome_threshold = 5
    # with_genome_5.auto_fill_code()
    #
    # with_auto_GT = copy.deepcopy(with_genome_3)
    # with_auto_GT.genome_threshold = "auto"
    # with_auto_GT.auto_fill_code()
    #
    # with_ss_and_G3 = copy.deepcopy(with_genome_3)
    # with_ss_and_G3.include_sample_quantity = True
    # with_ss_and_G3.auto_fill_code()
    #
    # with_ss_and_G4 = copy.deepcopy(with_genome_4)
    # with_ss_and_G4.include_sample_quantity = True
    # with_ss_and_G4.auto_fill_code()
    #
    # with_ss_and_G5 = copy.deepcopy(with_genome_5)
    # with_ss_and_G5.include_sample_quantity = True
    # with_ss_and_G5.auto_fill_code()
    #
    # with_median_diff = copy.deepcopy(with_ss_and_G3)
    # with_median_diff.include_mean_fitness = False
    # with_median_diff.include_diff_median = True
    # with_median_diff.auto_fill_code()
    #
    # with_median_diff_and_wasserstein = copy.deepcopy(with_median_diff)
    # with_median_diff_and_wasserstein.include_wasserstein_distance = True
    # with_median_diff_and_wasserstein.auto_fill_code()

    known_winner = PolishSearchSettings(code_name="known_winner",
                                        population_size=200,
                                        evaluation_budget=evaluation_budget,
                                        cluster_info_file_name=cluster_info_file_name,
                                        include_ps_len=True,
                                        include_sample_quantity=True,
                                        include_mean_fitness=True,
                                        include_atomicity=True,
                                        use_custom_atomicity=True,
                                        use_custom_sampling_operator=True,
                                        use_custom_mutation_operator=True,
                                        use_custom_crossover_operator=True,
                                        genome_threshold=3,
                                        )

    kw_withough_ps_len = copy.deepcopy(known_winner)
    kw_withough_ps_len.include_ps_len = False

    kw_without_sample_quantity = copy.deepcopy(known_winner)
    kw_without_sample_quantity.include_sample_quantity = False

    kw_without_mean_fitness = copy.deepcopy(known_winner)
    kw_without_mean_fitness.include_mean_fitness = False

    kw_without_atomicity = copy.deepcopy(known_winner)
    kw_without_atomicity.include_atomicity = False

    kw_without_sampling = copy.deepcopy(known_winner)
    kw_without_sampling.use_custom_sampling_operator = False

    kw_without_mutation = copy.deepcopy(known_winner)
    kw_without_mutation.use_custom_mutation_operator = False

    kw_without_crossover = copy.deepcopy(known_winner)
    kw_without_crossover.use_custom_crossover_operator = False

    kw_with_auto_gt = copy.deepcopy(known_winner)
    kw_with_auto_gt.genome_threshold = "auto"

    kw_with_g4 = copy.deepcopy(known_winner)
    kw_with_g4.genome_threshold = 4

    kw_with_g5 = copy.deepcopy(known_winner)
    kw_with_g5.genome_threshold = 5

    configs = [known_winner,
               kw_withough_ps_len, kw_without_sample_quantity, kw_without_mean_fitness, kw_without_atomicity,
               kw_without_sampling, kw_without_mutation, kw_without_crossover,
               kw_with_auto_gt, kw_with_g4, kw_with_g5]

    for config in configs:
        config.auto_fill_code()

    old_atomicity_metric = None
    if any(not settings.use_custom_atomicity for settings in configs):
        old_atomicity_metric = GlobalLinkageBasedOnMutualInformation()
        old_atomicity_metric.set_pRef(train_pRef)

    collected_data = []
    for config in configs:
        print(f"Running for settings {config.code_name}")
        settings_json = config.to_dict()
        try:
            result = search_for_pss_using_genome_threshold(train_session_data=train_pRef,
                                                           optimised_session_data=train_SPRef,
                                                           search_settings=config,
                                                           old_atomicity_metric=old_atomicity_metric)
            result_json = results_to_json(result, search_settings=config,
                                          test_session_data=test_SPRef)
            collected_data.append({"config": settings_json,
                                   "results": result_json})
        except Exception as e:
            collected_data.append({"config": settings_json,
                                   "error": repr(e)})

    destination_file = os.path.join(destination_path, "run_" + utils.get_formatted_timestamp() + "_" + repr(
        random.randrange(1000)) + ".json")
    with utils.open_and_make_directories(destination_file) as file:
        json.dump(collected_data, file)

    print("All done!")


#
#
# async def main():
#     tasks = [asyncio.to_thread(run) for _ in range(12)]
#     await asyncio.gather(*tasks)
#
# asyncio.run(main())
#


def main():
    for i in range(12):
        print(f"Iteration {i}")
        run()


main()
