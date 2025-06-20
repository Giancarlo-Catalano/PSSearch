#!/usr/bin/env python3
import sys

from Gian_experimental.NSGAIICustom.testing_in_vitro.SPRef import OptimisedSPref
from PolishSystem.read_data import get_pRef_from_vectors
from retail_forecasting_data_collection.configurations import configs
from retail_forecasting_data_collection.data_file_names import vector_path, fitness_values_path
from retail_forecasting_data_collection.running_the_GA import single_data_collection_run


def run_data_collection_using_seed(seed):
    original_pRef = get_pRef_from_vectors(name_of_vectors_file=vector_path,
                                          name_of_fitness_file=fitness_values_path,
                                          column_in_fitness_file=2)

    train_pRef, test_pRef = original_pRef.train_test_split(test_size=0.2, random_state=seed)

    optimised_train_SPref = OptimisedSPref.from_pRef(train_pRef)

    single_data_collection_run(train_pRef=train_pRef,
                               train_SPRef=optimised_train_SPref,
                               list_of_configs=configs,
                               extra_info ={"version": "testing_v0",
                                            "seed": seed})


def main():
    if len(sys.argv) > 1:
        try:
            number = int(sys.argv[1])
            run_data_collection_using_seed(number)
        except ValueError:
            print("Error: First argument must be an integer.")
    else:
        print("No command-line argument provided.")

main()