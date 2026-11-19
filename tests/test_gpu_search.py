import numpy as np
import pytest

from speech_vector_search import gpu_search, ranking

torch = pytest.importorskip('torch')
cuda = pytest.mark.skipif(not torch.cuda.is_available(),
    reason='CUDA is not available')


def assert_rows_match_ranking(scores, excluded, candidates, top_k):
    tensor = torch.from_numpy(scores)
    neighbours, neighbour_scores = gpu_search.k_nearest_neighbours_rows(
        tensor, top_k, excluded_indices=excluded)
    ranks, rank_scores = gpu_search.global_ranks_rows(tensor, candidates)
    for row, similarities in enumerate(scores):
        row_excluded = None if excluded is None else excluded[row]
        expected = ranking.k_nearest_neighbours(similarities, top_k,
            excluded_index=row_excluded)
        row_candidates = candidates[row] if candidates.ndim == 2 else candidates
        np.testing.assert_array_equal(neighbours[row], expected)
        np.testing.assert_array_equal(neighbour_scores[row],
            similarities[expected])
        np.testing.assert_array_equal(ranks[row],
            ranking.global_ranks(similarities, row_candidates))
        np.testing.assert_array_equal(rank_scores[row],
            similarities[row_candidates])
    assert neighbours.dtype == ranks.dtype == np.dtype(np.intp)


@pytest.mark.parametrize('rounding', [None, 1, 0])
def test_rows_match_ranking_with_random_scores_and_ties(rounding):
    rng = np.random.default_rng(7)
    for n_candidates in (2, 9, 62):
        scores = rng.normal(size=(5, n_candidates)).astype('float32')
        if rounding is not None: scores = np.round(scores, rounding)
        excluded = rng.integers(n_candidates, size=5)
        shared = rng.integers(n_candidates, size=7)
        per_row = rng.integers(n_candidates, size=(5, 4))
        for top_k in sorted({1, n_candidates // 2, n_candidates - 1}):
            assert_rows_match_ranking(scores, excluded, shared, top_k)
            assert_rows_match_ranking(scores, None, per_row, top_k)


def test_identical_scores_rank_by_index():
    scores = torch.zeros((2, 5))
    neighbours, _ = gpu_search.k_nearest_neighbours_rows(scores, 3,
        excluded_indices=1)
    np.testing.assert_array_equal(neighbours, [[0, 2, 3], [0, 2, 3]])
    ranks, _ = gpu_search.global_ranks_rows(scores, [4, 0, 2])
    np.testing.assert_array_equal(ranks, [[4, 0, 2], [4, 0, 2]])


def test_scorer_matches_index_scores():
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(20, 4))
    queries = rng.normal(size=(7, 4))
    scorer = gpu_search.GpuScorer(vectors, device='cpu')
    batches = list(scorer.cosine_batches(queries, batch_size=3))
    assert [start for start, _ in batches] == [0, 3, 6]
    scores = torch.cat([batch for _, batch in batches]).numpy()
    expected = (queries / np.linalg.norm(queries, axis=1, keepdims=True)
        @ (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).T)
    np.testing.assert_allclose(scores, expected, atol=1e-6)
    assert scores.dtype == np.float32 and len(scorer) == 20


def test_empty_candidates():
    ranks, scores = gpu_search.global_ranks_rows(torch.ones((2, 3)), [])
    assert ranks.shape == scores.shape == (2, 0)


def test_unavailable_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='CUDA'):
        gpu_search.GpuScorer(np.ones((2, 2)), device='cuda')


@pytest.mark.parametrize('scores', [torch.tensor([[0., float('nan')]]),
    torch.ones(3), np.ones((2, 2)), torch.ones((2, 2), dtype=torch.int64)])
def test_invalid_scores(scores):
    with pytest.raises(ValueError, match='scores'):
        gpu_search.global_ranks_rows(scores, [0])


@pytest.mark.parametrize('top_k', [0, True, 1.5, 4])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match='top_k'):
        gpu_search.k_nearest_neighbours_rows(torch.ones((2, 4)), top_k,
            excluded_indices=0)


@pytest.mark.parametrize('excluded', [True, -1, 4, [0], [0.5, 1.]])
def test_invalid_excluded_indices(excluded):
    with pytest.raises(ValueError, match='excluded_indices'):
        gpu_search.k_nearest_neighbours_rows(torch.ones((2, 4)), 1,
            excluded_indices=excluded)


@pytest.mark.parametrize('candidates', [[-1], [4], [[0], [1], [2]]])
def test_invalid_candidates(candidates):
    with pytest.raises(ValueError, match='candidate_indices'):
        gpu_search.global_ranks_rows(torch.ones((2, 4)), candidates)


def test_invalid_scorer_inputs():
    with pytest.raises(ValueError, match='vectors'):
        gpu_search.GpuScorer(np.ones(3), device='cpu')
    scorer = gpu_search.GpuScorer(np.ones((2, 3)), device='cpu')
    with pytest.raises(ValueError, match='batch_size'):
        next(scorer.cosine_batches(np.ones((1, 3)), batch_size=0))
    with pytest.raises(ValueError, match='dimension'):
        next(scorer.cosine_batches(np.ones((1, 2))))
    with pytest.raises(ValueError, match='finite'):
        next(scorer.cosine_batches(np.array([[np.nan, 1., 1.]])))


@cuda
def test_cuda_rows_match_ranking_on_cuda_scores():
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(500, 16))
    queries = rng.normal(size=(9, 16))
    scorer = gpu_search.GpuScorer(vectors)
    _, scores = next(scorer.cosine_batches(queries))
    neighbours, _ = gpu_search.k_nearest_neighbours_rows(scores, 30,
        excluded_indices=4)
    ranks, _ = gpu_search.global_ranks_rows(scores, np.arange(0, 500, 17))
    for row, similarities in enumerate(scores.cpu().numpy()):
        np.testing.assert_array_equal(neighbours[row],
            ranking.k_nearest_neighbours(similarities, 30, excluded_index=4))
        np.testing.assert_array_equal(ranks[row],
            ranking.global_ranks(similarities, np.arange(0, 500, 17)))
