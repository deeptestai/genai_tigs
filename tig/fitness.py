import numpy as np


def calculate_fitness(logit: np.ndarray, label: int) -> float:
    """Margin between the expected class and the best competing class.

    A negative value means the classifier no longer predicts ``label``.
    """
    best_two = np.argsort(-logit)[:2]
    competitor = best_two[1] if best_two[0] == label else best_two[0]
    return float(logit[label] - logit[competitor])
