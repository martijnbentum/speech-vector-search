import numpy as np


def l2_normalize(vector):
    '''normalize one vector.
    vector                  input vector
    '''
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.copy()
    return vector / norm


def l2_normalize_rows(vectors):
    '''normalize matrix rows.
    vectors                  input matrix
    '''
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
