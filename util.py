
import numpy as np


def confusion_matrix_normalizer(cm, strip_diagonal=True, normalize_rows=True, normalize_matrix=False):
    cm = cm.astype(np.float32)

    # Get rid of the diagonal. This allows to consider only the error-part of the conf-mat.
    if strip_diagonal:
        cm -= np.diag(cm) * np.eye(cm.shape[0])

    # Normalize each row. This allows to see what the distribution of error per class is.
    if normalize_rows:
        cm /= cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)

    # Normalize entire matrix. Allows to see the distribution of error overall.
    if normalize_matrix:
        cm /= cm.sum()

    return cm


def make_random_spots_cm(n=10, normalize=False):
    """
    Create a 10X10 binary matrix with n off-diagonal location with '1'
    :param n: int
        the number of spots
    :param normalize: bool
        normalize rows?
    :return:
    """
    cm = np.zeros((10, 10))
    for _ in range(n):
        while True:
            i, j = np.random.choice(range(10), 2, replace=False)
            if cm[i, j] == 0:
                cm[i, j] = 1
                break

    if normalize:
        cm = confusion_matrix_normalizer(cm, strip_diagonal=False, normalize_matrix=False, normalize_rows=True)

    return cm.astype(np.float32)

