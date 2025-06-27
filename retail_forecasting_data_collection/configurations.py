import copy

from initial_testing.data_collection_25_06_9.genome_threshold_search import PolishSearchSettings
from retail_forecasting_data_collection.data_file_names import cluster_info_path

evaluation_budget = 10000
population_size = 100

# known_winner = PolishSearchSettings(code_name="known_winner",
#                                     population_size=population_size,
#                                     evaluation_budget=evaluation_budget,
#                                     cluster_info_file_name=cluster_info_path,
#                                     include_ps_len=True,
#                                     include_sample_quantity=True,
#                                     include_mean_fitness=True,
#                                     include_atomicity=True,
#                                     use_custom_atomicity=True,
#                                     use_custom_sampling_operator=True,
#                                     use_custom_mutation_operator=True,
#                                     use_custom_crossover_operator=True,
#                                     genome_threshold=3,
#                                     )
#
# kw_withough_ps_len = copy.deepcopy(known_winner)
# kw_withough_ps_len.include_ps_len = False
#
# kw_without_sample_quantity = copy.deepcopy(known_winner)
# kw_without_sample_quantity.include_sample_quantity = False
#
# kw_without_mean_fitness = copy.deepcopy(known_winner)
# kw_without_mean_fitness.include_mean_fitness = False
#
# kw_without_atomicity = copy.deepcopy(known_winner)
# kw_without_atomicity.include_atomicity = False
#
# kw_without_sampling = copy.deepcopy(known_winner)
# kw_without_sampling.use_custom_sampling_operator = False
#
# kw_without_mutation = copy.deepcopy(known_winner)
# kw_without_mutation.use_custom_mutation_operator = False
#
# kw_without_crossover = copy.deepcopy(known_winner)
# kw_without_crossover.use_custom_crossover_operator = False
#
# kw_with_auto_gt = copy.deepcopy(known_winner)
# kw_with_auto_gt.genome_threshold = "auto"
#
# kw_with_g4 = copy.deepcopy(known_winner)
# kw_with_g4.genome_threshold = 4
#
# kw_with_g5 = copy.deepcopy(known_winner)
# kw_with_g5.genome_threshold = 5
#
# configs = [known_winner,
#            kw_withough_ps_len, kw_without_sample_quantity, kw_without_mean_fitness, kw_without_atomicity,
#            kw_without_sampling, kw_without_mutation, kw_without_crossover,
#            kw_with_auto_gt, kw_with_g4, kw_with_g5]


known_winner_without_gt = PolishSearchSettings(code_name="known_winner_without_gt",
                                               population_size=population_size,
                                               evaluation_budget=evaluation_budget,
                                               cluster_info_file_name=cluster_info_path,
                                               include_ps_len=True,
                                               include_sample_quantity=True,
                                               include_mean_fitness=True,
                                               include_atomicity=True,
                                               use_custom_atomicity=True,
                                               use_custom_sampling_operator=True,
                                               use_custom_mutation_operator=True,
                                               use_custom_crossover_operator=True,
                                               genome_threshold=None,
                                               )

kw_withough_ps_len_without_gt = copy.deepcopy(known_winner_without_gt)
kw_withough_ps_len_without_gt.include_ps_len = False

kw_without_sample_quantity_without_gt = copy.deepcopy(known_winner_without_gt)
kw_without_sample_quantity_without_gt.include_sample_quantity = False

kw_without_mean_fitness_without_gt = copy.deepcopy(known_winner_without_gt)
kw_without_mean_fitness_without_gt.include_mean_fitness = False

kw_without_atomicity_without_gt = copy.deepcopy(known_winner_without_gt)
kw_without_atomicity_without_gt.include_atomicity = False

kw_without_sampling_without_gt = copy.deepcopy(known_winner_without_gt)
kw_without_sampling_without_gt.use_custom_sampling_operator = False

kw_without_mutation_without_gt = copy.deepcopy(known_winner_without_gt)
kw_without_mutation_without_gt.use_custom_mutation_operator = False

kw_without_crossover_without_gt = copy.deepcopy(known_winner_without_gt)
kw_without_crossover_without_gt.use_custom_crossover_operator = False


configs = [known_winner_without_gt,
           kw_withough_ps_len_without_gt, kw_without_sample_quantity_without_gt, kw_without_mean_fitness_without_gt, kw_without_atomicity_without_gt,
           kw_without_sampling_without_gt, kw_without_mutation_without_gt, kw_without_crossover_without_gt]

for config in configs:
    config.auto_fill_code()