'''Exact selection and ranks over a shared similarity row.'''

from numbers import Integral

import numpy as np


def k_nearest_neighbours(similarities, top_k, excluded_index=None):
    '''Return nearest indices; equal scores use ascending candidate index.

    similarities:    finite one-dimensional score array
    top_k:           number of candidates to return
    excluded_index:  optional candidate to omit from neighbours only
    '''
    similarities = check_similarities(similarities)
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
    similarities = check_similarities(similarities)
    indices = check_candidate_indices(candidate_indices, len(similarities))
    return np.asarray([
        np.count_nonzero(similarities > similarities[i])
        + np.count_nonzero(similarities[:i] == similarities[i])
        for i in indices], dtype=np.intp)


def check_similarities(similarities):
    '''Return similarities as a finite one-dimensional float array.'''
    similarities = np.asarray(similarities)
    if similarities.ndim != 1:
        raise ValueError('similarities must be one-dimensional')
    # Ordering negates scores, which wraps or overflows for integers.
    if similarities.dtype.kind != 'f':
        raise ValueError('similarities must be floating-point values')
    if not np.isfinite(similarities).all():
        raise ValueError('similarities must contain finite values')
    return similarities


def check_candidate_indices(candidate_indices, n_candidates):
    '''Return non-negative in-bounds integer candidate indices.'''
    indices = np.asarray(candidate_indices)
    if indices.size == 0: return np.empty(0, dtype=np.intp)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise ValueError('candidate_indices must be one-dimensional integers')
    if indices.min() < 0 or indices.max() >= n_candidates:
        raise ValueError('candidate_indices are out of bounds')
    return indices.astype(np.intp)
