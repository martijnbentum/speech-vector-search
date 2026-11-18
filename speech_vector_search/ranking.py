'''Exact selection and ranks over a shared similarity row.'''

from numbers import Integral

import numpy as np


def k_nearest_neighbours(similarities, top_k, excluded_index=None):
    '''Return nearest indices; equal scores use ascending candidate index.

    similarities:    finite one-dimensional score array
    top_k:           number of candidates to return
    excluded_index:  optional candidate to omit from neighbours only
    '''
    similarities = np.asarray(similarities)
    if similarities.ndim != 1:
        raise ValueError('similarities must be one-dimensional')
    n_candidates = len(similarities)
    if excluded_index is not None:
        if isinstance(excluded_index, bool) or not isinstance(excluded_index,
            Integral):
            raise ValueError('excluded_index must be an integer')
        if not 0 <= excluded_index < n_candidates:
            raise ValueError('excluded_index is out of bounds')
    available = n_candidates - (excluded_index is not None)
    if (isinstance(top_k, bool) or not isinstance(top_k, Integral)
        or not 1 <= top_k <= available):
        raise ValueError('top_k must be between 1 and available candidates')
    # One extra candidate leaves top_k after exclusion, even at a tie.
    count = top_k + (excluded_index is not None)
    kth = n_candidates - count
    threshold = np.partition(similarities, kth)[kth]
    above = np.flatnonzero(similarities > threshold)
    ties = np.flatnonzero(similarities == threshold)[:count - len(above)]
    nearest = np.concatenate((above, ties))
    nearest = nearest[np.lexsort((nearest, -similarities[nearest]))]
    if excluded_index is not None:
        nearest = nearest[nearest != excluded_index]
    return nearest[:top_k]


def global_ranks(similarities, candidate_indices):
    '''Return zero-based global ranks without excluding any candidate.

    similarities:      finite one-dimensional score array
    candidate_indices: candidates to rank, in the requested output order
    Equal scores rank by ascending candidate index. Calculations retain
    the input scoring precision.
    '''
    similarities = np.asarray(similarities)
    if similarities.ndim != 1:
        raise ValueError('similarities must be one-dimensional')
    return np.asarray([
        np.count_nonzero(similarities > similarities[i])
        + np.count_nonzero(similarities[:i] == similarities[i])
        for i in candidate_indices], dtype=np.intp)
