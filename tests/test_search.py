import numpy as np

from speech_vector_search import metadata
from speech_vector_search import search


def _metadatas():
    return metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('a', 'word', [b'\xe1']),
        metadata.VectorMetadata('b', 'word', [b'\xe2']),
    ])


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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
    try:
        search.VectorIndex(vectors, items, backend='annoy')
    except ValueError as exc:
        assert 'unsupported backend' in str(exc)
        return
    raise AssertionError('expected ValueError for unsupported backend')


def test_vector_index_rejects_one_dimensional_vectors():
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
    ])
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
    ])
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
    items = metadata.VectorMetadatas([
        metadata.VectorMetadata('a', 'word', [b'\xe0']),
        metadata.VectorMetadata('b', 'word', [b'\xe1']),
    ])
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
