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
destination_path = r"C:\Users\gac8\PycharmProjects\PSSearch\initial_testing\data_collection_25_06_9\results\v4"


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

old_atomicity_metric = GlobalLinkageBasedOnMutualInformation()
old_atomicity_metric.set_pRef(train_pRef)

def run():


    print("Starting data collection")
    evaluation_budget = 5000

    baseline = PolishSearchSettings(code_name="[OS OM OC][L MF OA][GN]",
                                    population_size=100,
                                    evaluation_budget=evaluation_budget,
                                    cluster_info_file_name=cluster_info_file_name,
                                    include_ps_len=True,
                                    include_mean_fitness=True,
                                    include_atomicity=True,
                                    use_custom_atomicity=False,
                                    )

    with_new_atomicity = copy.deepcopy(baseline)
    with_new_atomicity.use_custom_atomicity = True
    with_new_atomicity.code_name = "[NS NM NC][L MF NA][GN]"

    with_genome_3 = copy.deepcopy(with_new_atomicity)
    with_genome_3.genome_threshold = 3
    with_genome_3.use_custom_sampling_operator = True
    with_genome_3.use_custom_mutation_operator = True
    with_genome_3.use_custom_crossover_operator = True
    with_genome_3.code_name = "[NS NM NC][L MF NA][G3]"

    with_genome_4 = copy.deepcopy(with_genome_3)
    with_genome_4.genome_threshold = 4
    with_genome_4.code_name = "[NS NM NC][L MF NA][G4]"


    with_genome_5 = copy.deepcopy(with_genome_3)
    with_genome_5.genome_threshold = 5
    with_genome_5.code_name = "[NS NM NC][L MF NA][G5]"

    with_ss = copy.deepcopy(with_genome_3)
    with_ss.include_sample_quantity = True
    with_ss.code_name = "[NS NM NC][L SS MF NA][G3]"

    with_median_diff = copy.deepcopy(with_ss)
    with_median_diff.include_mean_fitness = False
    with_median_diff.include_diff_median = True
    with_median_diff.code_name = "[NS NM NC][L SS DM NA][G3]"

    with_median_diff_and_wasserstein = copy.deepcopy(with_median_diff)
    with_median_diff_and_wasserstein.include_wasserstein_distance = True
    with_median_diff_and_wasserstein.code_name = "[NS NM NC][L SS DM WD NA][G3]"


    configs = [baseline,
               with_new_atomicity,
               with_genome_3, with_genome_4, with_genome_5,
               with_ss,
               with_median_diff,
               with_median_diff_and_wasserstein]


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
            collected_data.append({"confid": settings_json,
                                   "error": repr(e)})


    destination_file = os.path.join(destination_path, "run_"+utils.get_formatted_timestamp()+"_"+repr(random.randrange(1000))+".json")
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
