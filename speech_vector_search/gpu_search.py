'''Optional GPU add-on: cosine scores, neighbours and ranks with torch.

Results follow ranking.py: equal scores rank by ascending candidate index.
Scores are float32 and may differ slightly from CPU scores, so ranks of
near-equal candidates can differ between backends. Install the gpu extra.
'''

from numbers import Integral

import numpy as np

from speech_vector_search import ranking, util_math


class GpuScorer:
    '''Device copy of candidate vectors for batched cosine scoring.

    vectors:  candidate matrix, for example VectorIndex.vectors
    device:   torch device; tests may use 'cpu'
    '''

    def __init__(self, vectors, device='cuda'):
        torch = import_torch()
        self.device = check_device(torch, device)
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError('vectors must be a finite two-dimensional array')
        vectors = util_math.l2_normalize_rows(vectors).astype('float32')
        self.vectors = torch.from_numpy(vectors).to(self.device)

    def __len__(self):
        return self.vectors.shape[0]

    def cosine_batches(self, query_vectors, batch_size=256):
        '''Yield (start, scores) batches as float32 device tensors.

        query_vectors:  query matrix, normalized before scoring
        batch_size:     maximum number of score rows held at once
        '''
        torch = import_torch()
        if (isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral) or batch_size < 1):
            raise ValueError('batch_size must be a positive integer')
        query_vectors = np.asarray(query_vectors)
        if query_vectors.ndim != 2:
            raise ValueError('query vectors must be two-dimensional')
        if query_vectors.shape[1] != self.vectors.shape[1]:
            m = f'query vector dimension {query_vectors.shape[1]} '
            m += f'does not match index {self.vectors.shape[1]}'
            raise ValueError(m)
        for start in range(0, len(query_vectors), batch_size):
            queries = query_vectors[start:start + batch_size]
            if not np.isfinite(queries).all():
                raise ValueError('query vectors must contain finite values')
            queries = util_math.l2_normalize_rows(queries).astype('float32')
            queries = torch.from_numpy(queries).to(self.device)
            yield start, queries @ self.vectors.T


def k_nearest_neighbours_rows(scores, top_k, excluded_indices=None):
    '''Return (indices, similarities) of each score row's nearest candidates.

    scores:            finite B x N score tensor
    top_k:             number of candidates to return per row
    excluded_indices:  optional candidate per row (or one for all rows) to
                       omit from that row's neighbours
    Returns two B x top_k NumPy arrays, matching ranking.k_nearest_neighbours.
    '''
    torch = import_torch()
    check_scores(torch, scores)
    n_rows, n_candidates = scores.shape
    excluded = check_excluded(excluded_indices, n_rows, n_candidates)
    available = n_candidates - (excluded is not None)
    if (isinstance(top_k, bool) or not isinstance(top_k, Integral)
        or not 1 <= top_k <= available):
        raise ValueError('top_k must be between 1 and available candidates')
    count = top_k + (excluded is not None)
    indices = _top_indices(torch, scores, count)
    if excluded is not None:
        excluded = torch.from_numpy(excluded).to(scores.device)
        keep = (indices != excluded[:, None]).to(torch.int8)
        order = torch.sort(keep, dim=1, descending=True, stable=True).indices
        indices = torch.gather(indices, 1, order[:, :top_k])
    similarities = torch.gather(scores, 1, indices)
    return indices.cpu().numpy().astype(np.intp), similarities.cpu().numpy()


def global_ranks_rows(scores, candidate_indices):
    '''Return (ranks, similarities) of candidates in each score row.

    scores:             finite B x N score tensor
    candidate_indices:  m candidates shared by all rows, or B x m per row
    Ranks are zero-based over all N candidates, without exclusions, and
    match ranking.global_ranks. Returns two B x m NumPy arrays.
    '''
    torch = import_torch()
    check_scores(torch, scores)
    n_rows, n_candidates = scores.shape
    indices = check_candidate_matrix(candidate_indices, n_rows, n_candidates)
    if indices.shape[1] == 0:
        return np.empty(indices.shape, np.intp), np.empty(indices.shape,
            np.float32)
    indices = torch.from_numpy(indices).to(scores.device)
    similarities = torch.gather(scores, 1, indices)
    ordered = torch.sort(scores, dim=1).values
    greater = torch.searchsorted(ordered, similarities, right=True)
    greater = n_candidates - greater
    at_least = n_candidates - torch.searchsorted(ordered, similarities)
    ranks = greater + _equal_before(torch, scores, indices, similarities,
        at_least - greater)
    return ranks.cpu().numpy().astype(np.intp), similarities.cpu().numpy()


def import_torch():
    '''Import torch, which the gpu extra installs.'''
    try:
        import torch
    except ImportError:
        m = 'gpu_search requires torch; install speech_vector_search[gpu]'
        raise ImportError(m) from None
    return torch


def check_device(torch, device):
    '''Return a torch device, refusing CUDA when it is unavailable.'''
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available on this machine')
    return device


def check_scores(torch, scores):
    '''Require a finite two-dimensional floating-point score tensor.'''
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise ValueError('scores must be a two-dimensional torch tensor')
    if not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError('scores must contain finite floating-point values')


def check_excluded(excluded_indices, n_rows, n_candidates):
    '''Return one excluded candidate per row, or None.'''
    if excluded_indices is None: return None
    if isinstance(excluded_indices, Integral) and not isinstance(
        excluded_indices, bool):
        excluded_indices = [excluded_indices] * n_rows
    excluded = np.asarray(excluded_indices)
    if excluded.shape != (n_rows,) or excluded.dtype.kind not in 'iu':
        m = 'excluded_indices must be an integer or one integer per row'
        raise ValueError(m)
    if n_rows and (excluded.min() < 0 or excluded.max() >= n_candidates):
        raise ValueError('excluded_indices are out of bounds')
    return excluded.astype(np.int64)


def check_candidate_matrix(candidate_indices, n_rows, n_candidates):
    '''Return B x m int64 candidates from shared or per-row indices.'''
    indices = np.asarray(candidate_indices)
    if indices.ndim == 2 and indices.shape[0] == n_rows:
        flat = ranking.check_candidate_indices(indices.ravel(), n_candidates)
        return flat.reshape(indices.shape).astype(np.int64)
    if indices.ndim == 2:
        raise ValueError('candidate_indices must have one row per score row')
    shared = ranking.check_candidate_indices(indices, n_candidates)
    return np.tile(shared.astype(np.int64), (n_rows, 1))


def _top_indices(torch, scores, count):
    '''Return count best indices per row by score, then ascending index.'''
    threshold = torch.topk(scores, count, dim=1).values[:, -1:]
    selected = scores > threshold
    ties = scores == threshold
    needed = count - selected.sum(1, keepdim=True)
    ambiguous = (ties.sum(1, keepdim=True) > needed).flatten()
    if ambiguous.any():
        # Keep only the lowest-index ties needed to fill these rows.
        rows = ambiguous.nonzero().flatten()
        first = torch.cumsum(ties[rows].to(torch.int32), dim=1)
        ties[rows] &= first <= needed[rows]
    selected |= ties
    # nonzero lists indices in ascending order within each row.
    indices = selected.nonzero()[:, 1].reshape(len(scores), count)
    values = torch.gather(scores, 1, indices)
    order = torch.sort(values, dim=1, descending=True, stable=True).indices
    return torch.gather(indices, 1, order)


def _equal_before(torch, scores, indices, similarities, n_equal,
    chunk_size=256):
    '''Count, per candidate, equal scores at lower candidate indices.'''
    extra = torch.zeros_like(indices)
    # Only candidates whose score occurs more than once need a prefix count.
    rows, columns = (n_equal > 1).nonzero(as_tuple=True)
    positions = torch.arange(scores.shape[1], device=scores.device)
    for start in range(0, len(rows), chunk_size):
        r = rows[start:start + chunk_size]
        c = columns[start:start + chunk_size]
        before = positions[None, :] < indices[r, c][:, None]
        equal = scores[r] == similarities[r, c][:, None]
        extra[r, c] = (equal & before).sum(1)
    return extra
