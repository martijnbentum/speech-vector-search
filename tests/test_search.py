from unittest import mock

import numpy as np
import pytest

from speech_vector_search import metadata
from speech_vector_search import search


def _metadatas():
    return metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('a', 'word', [b'\xe1']),
        metadata.VectorMetadata('b', 'word', [b'\xe2']),
    ], name='demo')


def _pair_metadatas():
    return metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ], name='demo')


def test_brute_force_query_by_index_returns_expected_neighbours():
    vectors = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])
    index = search.VectorIndex(vectors, _metadatas(), backend='brute_force')
    result = index.query_by_index(0, top_k=2)
    assert result['indices'].tolist() == [0, 1]
    assert result['metadatas'][1].label == 'a'


def test_faiss_query_by_index_returns_expected_neighbours():
    vectors = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])
    index = search.VectorIndex(vectors, _metadatas(), backend='faiss')
    result = index.query_by_index(0, top_k=2)
    assert result['indices'].tolist() == [0, 1]
    assert result['metadatas'][1].label == 'a'


def test_faiss_queries_returns_one_result_per_query():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='faiss')
    results = index.queries(np.array([[1.0, 0.0], [0.0, 1.0]]), top_k=1)
    assert len(results) == 2
    assert results[0]['indices'].tolist() == [0]
    assert results[1]['indices'].tolist() == [1]


def test_query_rejects_non_vector_input():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    try:
        index.query([[1.0, 0.0]])
    except ValueError as exc:
        assert 'vector must be 1D' in str(exc)
        return
    raise AssertionError('expected ValueError for non-1D query')


def test_query_rejects_wrong_embedding_dimension():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    try:
        index.query(np.array([1.0, 0.0, 0.0]))
    except ValueError as exc:
        assert 'does not match index' in str(exc)
        return
    raise AssertionError('expected ValueError for wrong query dimension')


def test_queries_rejects_one_dimensional_input():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    try:
        index.queries(np.array([1.0, 0.0]))
    except ValueError as exc:
        assert 'vectors must be 2D' in str(exc)
        return
    raise AssertionError('expected ValueError for one-dimensional query matrix')


def test_queries_rejects_wrong_embedding_dimension():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    try:
        index.queries(np.array([[1.0, 0.0, 0.0]]))
    except ValueError as exc:
        assert 'does not match index' in str(exc)
        return
    raise AssertionError('expected ValueError for wrong embedding dimension')


def test_queries_returns_one_result_per_query():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    results = index.queries(np.array([[1.0, 0.0], [0.0, 1.0]]), top_k=1)
    assert len(results) == 2
    assert results[0]['indices'].tolist() == [0]
    assert results[1]['indices'].tolist() == [1]


def test_query_by_index_rejects_out_of_bounds_index():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    try:
        index.query_by_index(2)
    except ValueError as exc:
        assert 'out of bounds' in str(exc)
        return
    raise AssertionError('expected ValueError for out-of-bounds index')


def test_query_by_index_rejects_negative_index():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    try:
        index.query_by_index(-1)
    except ValueError as exc:
        assert 'must be non-negative' in str(exc)
        return
    raise AssertionError('expected ValueError for negative index')


def test_check_top_k_clamps_to_index_size(capsys):
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    result = index.query_by_index(0, top_k=10)
    captured = capsys.readouterr()
    assert 'using 2 as top_k' in captured.out
    assert len(result['indices']) == 2


def test_check_top_k_clamps_to_index_size_for_faiss(capsys):
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='faiss')
    result = index.query_by_index(0, top_k=10)
    captured = capsys.readouterr()
    assert 'using 2 as top_k' in captured.out
    assert len(result['indices']) == 2


def test_result_rejects_invalid_backend_indices():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    try:
        index._result(np.array([0.0]), np.array([5]))
    except ValueError as exc:
        assert 'indices out of bounds' in str(exc)
        return
    raise AssertionError('expected ValueError for invalid backend indices')


def test_result_returns_metadata_for_valid_indices():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.VectorIndex(vectors, items, backend='brute_force')
    result = index._result(np.array([0.9, 0.8]), np.array([0, 1]))
    assert np.allclose(result['scores'], [0.9, 0.8])
    assert result['indices'].tolist() == [0, 1]
    assert [item.label for item in result['metadatas']] == ['a', 'b']


def test_vector_index_rejects_unsupported_backend():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    try:
        search.VectorIndex(vectors, items, backend='annoy')
    except ValueError as exc:
        assert 'unsupported backend' in str(exc)
        return
    raise AssertionError('expected ValueError for unsupported backend')


def test_vector_index_rejects_one_dimensional_vectors():
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
    ], name='demo')
    try:
        search.VectorIndex(np.array([1.0, 0.0]), items, backend='brute_force')
    except Exception as exc:
        assert 'axis 1 is out of bounds' in str(exc)
        return
    raise AssertionError('expected failure for one-dimensional vectors')


def test_vector_index_rejects_mismatched_metadata_length():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
    ], name='demo')
    try:
        search.VectorIndex(vectors, items, backend='brute_force')
    except ValueError as exc:
        assert 'must match number of metadatas' in str(exc)
        return
    raise AssertionError('expected ValueError for mismatched metadata length')


def test_vector_index_rejects_non_vector_metadatas():
    vectors = np.array([[1.0, 0.0]])
    try:
        search.VectorIndex(vectors, [{'label': 'a'}], backend='brute_force')
    except ValueError as exc:
        assert 'VectorMetadatas instance' in str(exc)
        return
    raise AssertionError('expected ValueError for invalid metadatas type')


def test_build_index_returns_vector_index():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    items = _pair_metadatas()
    index = search.build_index(vectors, items, backend='brute_force')
    assert isinstance(index, search.VectorIndex)


def test_check_single_vector_rejects_two_dimensional_input():
    try:
        search.check_single_vector(np.array([[1.0, 0.0]]))
    except ValueError as exc:
        assert 'vector must be 1D' in str(exc)
        return
    raise AssertionError('expected ValueError for two-dimensional single vector')


def test_check_multiple_vectors_rejects_one_dimensional_input():
    try:
        search.check_multiple_vectors(np.array([1.0, 0.0]))
    except ValueError as exc:
        assert 'vectors must be 2D' in str(exc)
        return
    raise AssertionError(
        'expected ValueError for one-dimensional multiple-vectors input')


@pytest.mark.parametrize('backend', ['faiss', 'brute_force'])
def test_cosine_batches_match_scoring_precision_and_bound_rows(backend):
    vectors = np.array([[1., 2.], [0., 0.], [-2., 1.]])
    index = search.VectorIndex(vectors, _metadatas(), backend=backend)
    queries = np.array([[3., 1.], [0., 0.], [-1., 2.], [1., -2.]])
    dtype = np.float32 if backend == 'faiss' else np.float64
    normalized = search.util_math.l2_normalize_rows(queries).astype(dtype)
    expected = normalized @ index.vectors.astype(dtype).T
    batches = list(index.cosine_batches(queries, batch_size=2))
    assert [start for start, _ in batches] == [0, 2]
    for start, scores in batches:
        assert scores.shape == (2, 3)
        assert scores.dtype == dtype
        np.testing.assert_array_equal(scores, expected[start:start + 2])
    assert list(index.cosine_batches(np.empty((0, 2)))) == []


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5])
def test_cosine_batches_reject_invalid_batch_size(batch_size):
    index = search.VectorIndex(np.eye(3, 2), _metadatas())
    with pytest.raises(ValueError, match='batch_size'):
        list(index.cosine_batches(np.ones((1, 2)), batch_size=batch_size))


@pytest.mark.parametrize('queries', [np.ones(2), np.ones((1, 3)),
    np.array([[np.nan, 0.]]), np.array([[np.inf, 0.]])])
def test_cosine_batches_reject_invalid_queries(queries):
    index = search.VectorIndex(np.eye(3, 2), _metadatas())
    with pytest.raises(ValueError):
        list(index.cosine_batches(queries))


@pytest.mark.parametrize('backend', ['faiss', 'brute_force'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_vector_index_rejects_nonfinite_candidates_at_construction(backend,
    value):
    vectors = np.eye(3, 2)
    vectors[0, 0] = value
    with np.errstate(invalid='ignore'):
        with pytest.raises(ValueError, match='finite'):
            search.VectorIndex(vectors, _metadatas(), backend=backend)


def test_faiss_index_rejects_float32_conversion_overflow():
    vectors = np.array([[np.finfo(np.float64).max, 0.]])
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError, match='finite float32'):
            search.FaissIndex(vectors)


@pytest.mark.parametrize('backend', ['faiss', 'brute_force'])
def test_repeated_cosine_calls_do_not_scan_or_cast_candidates(backend):
    index = search.VectorIndex(np.eye(3, 2), _metadatas(), backend=backend)
    queries = np.array([[1., 2.]])
    candidates = index.backend.vectors
    expected = (search.util_math.l2_normalize_rows(queries)
        .astype(candidates.dtype) @ candidates.T)
    with mock.patch.object(search.np, 'isfinite', wraps=np.isfinite) as check:
        with mock.patch.object(search.np, 'asarray', wraps=np.asarray) as cast:
            for _ in range(2):
                batches = list(index.cosine_batches(queries))
                np.testing.assert_array_equal(batches[0][1], expected)
            assert all(call.args[0].shape == queries.shape
                for call in check.call_args_list)
            assert all(call.args[0] is not index.vectors
                and call.args[0] is not candidates
                for call in cast.call_args_list)
    assert index.backend.vectors is candidates
