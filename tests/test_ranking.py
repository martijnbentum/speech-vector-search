import unittest

import numpy as np
import pytest

from speech_vector_search import ranking


class RankingTests(unittest.TestCase):
    def assert_matches_full_sort(self, scores):
        n_tokens = len(scores)
        n_types = 2 if n_tokens % 2 == 0 else 1
        token_types = np.arange(n_tokens) % n_types
        order = sorted(range(n_tokens), key=lambda i: (-scores[i], i))
        for token_index in range(n_tokens):
            same = np.flatnonzero(token_types == token_types[token_index])
            for top_k in sorted({1, n_tokens // 2, n_tokens - 1}):
                with self.subTest(token_index=token_index, top_k=top_k):
                    actual_neighbours = ranking.k_nearest_neighbours(
                        scores, top_k, excluded_index=token_index)
                    actual_ranks = ranking.global_ranks(scores, same)
                    neighbours = [i for i in order if i != token_index][:top_k]
                    np.testing.assert_array_equal(actual_neighbours, neighbours)
                    np.testing.assert_array_equal(actual_ranks,
                        [order.index(int(i)) for i in same])
                    self.assertEqual(actual_ranks.dtype, np.dtype(np.intp))

    def test_random_scores_match_full_sort_with_31_tokens_per_type(self):
        rng = np.random.default_rng(42)
        for dtype in ('float32', 'float64'):
            for count in (2, 9, 62):
                scores = rng.normal(size=count).astype(dtype)
                self.assert_matches_full_sort(scores)
                self.assert_matches_full_sort(np.round(scores, 1))

    def test_boundary_ties_identical_scores_and_negative_scores(self):
        for scores in ([.9, .8, .8, .8, .8, -.1], [0.] * 6,
            [-.2, -.1, -.1, -.8, -.4, -.1]):
            self.assert_matches_full_sort(np.asarray(scores))

    def test_float64_distinctions_become_ties_after_float32_conversion(self):
        scores = np.array([1., np.nextafter(1., 2.),
            np.nextafter(1., 0.), 1.], dtype='float64')
        np.testing.assert_array_equal(
            ranking.global_ranks(scores, range(4)), [1, 0, 3, 2])
        np.testing.assert_array_equal(
            ranking.global_ranks(scores.astype('float32'), range(4)),
            [0, 1, 2, 3])


def test_neighbours_without_exclusion_and_requested_rank_order():
    scores = np.array([0.5, 1., 1., -1.])
    np.testing.assert_array_equal(
        ranking.k_nearest_neighbours(scores, 4), [1, 2, 0, 3])
    np.testing.assert_array_equal(
        ranking.global_ranks(scores, [3, 1, 0, 1]), [3, 0, 2, 0])
    assert ranking.global_ranks(scores, []).shape == (0,)


@pytest.mark.parametrize('top_k', [0, -1, True, 1.5, 5])
def test_invalid_neighbour_count(top_k):
    with pytest.raises(ValueError, match='top_k'):
        ranking.k_nearest_neighbours(np.ones(4), top_k)


def test_exclusion_leaves_only_available_candidates():
    scores = np.ones(4)
    np.testing.assert_array_equal(
        ranking.k_nearest_neighbours(scores, 3, excluded_index=1), [0, 2, 3])
    with pytest.raises(ValueError, match='top_k'):
        ranking.k_nearest_neighbours(scores, 4, excluded_index=1)


@pytest.mark.parametrize('excluded_index', [True, False, 1.0, 1.5, -1, 4])
def test_invalid_excluded_index(excluded_index):
    with pytest.raises(ValueError, match='excluded_index'):
        ranking.k_nearest_neighbours(np.ones(4), 2,
            excluded_index=excluded_index)


def test_numpy_integer_excluded_index():
    np.testing.assert_array_equal(
        ranking.k_nearest_neighbours(np.ones(4), 3,
            excluded_index=np.int64(1)), [0, 2, 3])
