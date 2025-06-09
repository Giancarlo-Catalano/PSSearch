from collections import defaultdict
from typing import Callable

import numpy as np

from BenchmarkProblems.NK import NK
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core.PRef import PRef
from Core.PSMetric.Linkage.BivariateLinkage import BivariateLinkage
from Core.PSMetric.Linkage.ValueSpecificMutualInformation import SolutionSpecificMutualInformation
from Core.get_pRef import get_pRef_from_metaheuristic


class FinalLinkage:
    univariate_win_tables: defaultdict
    bivariate_win_tables: defaultdict
    pRef: PRef

    def __init__(self):
        super().__init__()

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.univariate_win_tables, self.bivariate_win_tables = self.get_tables()

    def get_tables(self):
        univariate_win_tables = defaultdict(int)  # the key is var, winner_val, lower_val
        bivariate_win_tables = defaultdict(
            int)  # the key is var_a, var_a, winner_val_a, winner_val_b, loser_val_a, loser_val_b

        for index_1, solution_1 in enumerate(self.pRef.full_solution_matrix):
            fitness_1 = self.pRef.fitness_array[index_1]
            for index_2, solution_2 in enumerate(self.pRef.full_solution_matrix[index_1 + 1:], start=index_1 + 1):
                diff: np.ndarray = solution_1 != solution_2

                diff_count = np.sum(diff)
                if diff_count > 2:
                    continue

                fitness_2 = self.pRef.fitness_array[index_2]
                if fitness_2 == fitness_1:
                    continue
                winner, loser = (solution_1, solution_2) if fitness_1 > fitness_2 else (solution_2, solution_1)

                diff_positions = [index for index, is_different in enumerate(diff) if is_different]
                if diff_count == 1:
                    var = diff_positions[0]
                    winner_val = winner[var]
                    loser_val = loser[var]
                    univariate_win_tables[var, winner_val, loser_val] += 1

                    # question: shouldn't we also update bivariate_win_tables with every other variable, paired with var?
                elif diff_count == 2:
                    var_a, var_b = diff_positions
                    winner_val_a = winner[var_a]
                    winner_val_b = winner[var_b]
                    loser_val_a = loser[var_a]
                    loser_val_b = loser[var_b]
                    bivariate_win_tables[var_a, var_b, winner_val_a, winner_val_b, loser_val_a, loser_val_b] += 1

        return univariate_win_tables, bivariate_win_tables

    def score_probability_of_train_vs_test(self, p_train, p_test):
        return p_test ** 2 + p_train * (1 - 2 * p_test)

    def get_univariate_predictions(self):
        predictions = defaultdict(float)

        for key in self.univariate_win_tables:
            var, win_val, lose_val = key
            swapped_key = var, lose_val, win_val
            count_wins = self.univariate_win_tables[key]
            count_losses = self.univariate_win_tables[swapped_key] if swapped_key in self.univariate_win_tables else 0
            if count_losses + count_wins == 0:
                print(f"Not enough data for {key}")
                continue
            prediction = count_wins / (count_wins + count_losses)
            predictions[key] = prediction
            predictions[swapped_key] = 1 - prediction
        return predictions

    def get_bivariate_predictions(self):
        predictions = defaultdict(float)

        for key in self.bivariate_win_tables:
            var_a, var_b, winner_val_a, winner_val_b, loser_val_a, loser_val_b = key
            swapped_key = var_a, var_b, loser_val_a, loser_val_b, winner_val_a, winner_val_b
            count_wins = self.bivariate_win_tables[key]
            count_losses = self.bivariate_win_tables[swapped_key] if swapped_key in self.bivariate_win_tables else 0
            if count_losses + count_wins == 0:
                print(f"Not enough data for {key}")
                continue
            prediction = count_wins / (count_wins + count_losses)
            predictions[key] = prediction
            predictions[swapped_key] = 1 - prediction
        return predictions

    def get_error_for_bivariate_predictions(self,
                                            univariate_predictions: defaultdict,
                                            bivariate_predictions: defaultdict,
                                            error_aggregation: Callable):

        errors_for_each_pair = defaultdict(list)
        for key in bivariate_predictions:
            p_test = bivariate_predictions[key]
            var_a, var_b, winner_val_a, winner_val_b, loser_val_a, loser_val_b = key

            key_a = (var_a, winner_val_a, loser_val_a)
            key_b = (var_b, winner_val_b, loser_val_b)
            if key_a not in univariate_predictions or key_b not in univariate_predictions:
                print(f"{key} could not have a prediction because the univariate components are not present")
                continue

            prediction_a = univariate_predictions[key_a]
            prediction_b = univariate_predictions[key_b]

            p_train = prediction_a * prediction_b

            error = self.score_probability_of_train_vs_test(p_train, p_test)
            errors_for_each_pair[frozenset({var_a, var_b})].append(error)

        n = self.pRef.search_space.amount_of_parameters
        linkage_table = np.zeros((n, n), dtype=float)
        for key, errors in errors_for_each_pair.items():
            average_error = error_aggregation(errors) if len(errors) > 0 else -1
            i, j = key
            linkage_table[i, j] = average_error

        linkage_table += linkage_table.T
        return linkage_table


def run():
    problem = NK.random(16, 3)
    pRef = get_pRef_from_metaheuristic(problem=problem,
                                       sample_size=10000,
                                       which_algorithm="GA",
                                       unique=True,
                                       verbose=True)

    print(f"The pRef has size {pRef.sample_size}")

    linkage_metric = FinalLinkage()
    linkage_metric.set_pRef(pRef)
    univariate_predictions = linkage_metric.get_univariate_predictions()
    bivariate_predictions = linkage_metric.get_bivariate_predictions()

    min_linkage_table = linkage_metric.get_error_for_bivariate_predictions(univariate_predictions,
                                                                       bivariate_predictions,
                                                                       error_aggregation = min)

    max_linkage_table = linkage_metric.get_error_for_bivariate_predictions(univariate_predictions,
                                                                           bivariate_predictions,
                                                                           error_aggregation=max)

    avg_linkage_table = linkage_metric.get_error_for_bivariate_predictions(univariate_predictions,
                                                                           bivariate_predictions,
                                                                           error_aggregation=np.average)


    print("Hello!")

run()