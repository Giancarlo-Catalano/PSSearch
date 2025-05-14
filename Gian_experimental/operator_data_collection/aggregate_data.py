import json
from os.path import isfile
import os
from os import listdir

# the intention is to have the all the result files in one folder, and analyse them together.

main_dir = r"C:\Users\gac8\PycharmProjects\PSSearch\Gian_experimental\operator_data_collection\V2"
files = listdir(main_dir)
files = [os.path.join(main_dir, file) for file in files]
files = [file for file in files if isfile(file) if file.endswith(".json")]


def get_data_for_file(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data["results"]


metrics = ["sample_sizes_test", "consistencies_test", "ones"]


def only_significant_pss(data_for_method):
    is_acceptable = [p_value < 0.05 and ones > 0 for p_value, ones
                     in
                     zip(data_for_method["run_result"]["consistencies_train"], data_for_method["run_result"]["ones"])]

    def filter_by_acceptable(items):
        return [item for item, accept in zip(items, is_acceptable) if accept]

    return {"method": data_for_method["method"],
            "run_result": {metric_name: filter_by_acceptable(items)
                            for metric_name, items in data_for_method["run_result"].items()}
            }


data_for_all_files = [only_significant_pss(results_for_method)
                      for file in files
                      for results_for_method in get_data_for_file(file)]

all_methods = {results_for_method["method"]
               for results_for_method in data_for_all_files}

aggregated = {method: {metric: []
                       for metric in metrics}
              for method in all_methods}

for data_for_a_method in data_for_all_files:
    method = data_for_a_method["method"]
    for metric in metrics:
        aggregated[method][metric].extend(data_for_a_method["run_result"][metric])



print(aggregated)










