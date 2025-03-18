from typing import Optional, Callable

from Core.FullSolution import FullSolution
from Core.PS import PS
from PolishSystem.PolishPSSearchTask import find_ps_in_polish_solution
from SimplifiedSystem.PSSearchSettings import PSSearchSettings
from SimplifiedSystem.search_methods import get_unexplained_parts_of_solution


def search_local_polish_ps(solution_to_explain: FullSolution,
                           search_settings: PSSearchSettings,
                           objectives: list[Callable],
                           to_avoid: Optional[list[PS]] = None) -> [PS]:
    unexplained_vars = get_unexplained_parts_of_solution(solution_to_explain, [] if to_avoid is None else to_avoid)
    return find_ps_in_polish_solution(to_explain=solution_to_explain,
                                      ps_budget=search_settings.ps_search_budget,
                                      culling_method=search_settings.culling_method,
                                      population_size=search_settings.ps_search_population,
                                      metrics_functions=objectives,
                                      unexplained_mask=unexplained_vars,
                                      proportion_unexplained_that_needs_used=search_settings.proportion_unexplained_that_needs_used,
                                      proportion_used_that_should_be_unexplained=search_settings.proportion_used_that_should_be_unexplained,
                                      sampling_operator=search_settings.sampling_operator,
                                      mutation_operator=search_settings.mutation_operator,
                                      crossover_operator=search_settings.crossover_operator,

                                      verbose=search_settings.verbose)
