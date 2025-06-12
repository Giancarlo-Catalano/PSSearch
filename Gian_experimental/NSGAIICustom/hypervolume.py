from typing import Iterable

import numpy as np
from deap.tools import hv

from Gian_experimental.NSGAIICustom.NSGAIICustom import EvaluatedNCSolution


def calculate_hypervolume(results: Iterable[EvaluatedNCSolution]) -> float:
    # Define a reference point (must be worse than all points in every objective)

    metrics = np.array([solution.fitnesses for solution in results])
    reference_point = np.max(metrics, axis=0)

    # Compute hypervolume
    volume = hv.hypervolutme(metrics, reference_point)

    return volume
