import copy
import json
import os
import asyncio
import random
from typing import Optional

import utils
from Core.PSMetric.Linkage.ValueSpecificMutualInformation import GlobalLinkageBasedOnMutualInformation
from Gian_experimental.NSGAIICustom.testing_in_vitro.SPRef import OptimisedSPref
from PolishSystem.read_data import get_pRef_from_vectors
from initial_testing.data_collection_25_06_9.genome_threshold_search import PolishSearchSettings, \
    search_for_pss_using_genome_threshold, results_to_json


def pss_to_json(list_of_pss):
    return [{"pattern": list(ps.solution),
             "threshold": int(ps.solution.genome_threshold) if ps.solution.genome_threshold is not None else None}
            for ps in list_of_pss]

import sys

def print_to_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)



def single_data_collection_run(train_pRef,
                               train_SPRef,
                               list_of_configs,
                               extra_info: dict,
                               save_to_folder: Optional[str]):

    old_atomicity_metric = None
    if any(not settings.use_custom_atomicity for settings in list_of_configs):
        old_atomicity_metric = GlobalLinkageBasedOnMutualInformation()
        old_atomicity_metric.set_pRef(train_pRef)

    collected_data = []
    for config in list_of_configs:
        print_to_error(f"Running for settings {config.code_name}")
        settings_json = config.to_dict()
        try:
            result = search_for_pss_using_genome_threshold(train_session_data=train_pRef,
                                                           optimised_session_data=train_SPRef,
                                                           search_settings=config,
                                                           old_atomicity_metric=old_atomicity_metric)
            result_json = pss_to_json(result)
            collected_data.append({"config": settings_json,
                                   "results": result_json})
        except Exception as e:
            collected_data.append({"config": settings_json,
                                   "error": repr(e)})
            raise e

    to_output = {
                    "extra_info": extra_info,
                    "data": collected_data,
                 }

    if save_to_folder is None:
        print(json.dumps(to_output))
    else:
        file_name = os.path.join(save_to_folder, f"result_"+utils.get_formatted_timestamp()+".json")
        with utils.open_and_make_directories(file_name) as file:
            json.dump(to_output, file)

