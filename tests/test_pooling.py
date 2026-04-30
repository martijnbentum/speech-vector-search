import numpy as np

from speech_vector_search import util_math


def test_pool_frames_mean():
    frames = np.array([[1.0, 3.0], [3.0, 5.0]])
    pooled = util_math.pool_frames(frames)
    assert np.allclose(pooled, [2.0, 4.0])


def test_pool_frames_rejects_unknown_method():
    frames = np.array([[1.0, 2.0]])
    try:
        util_math.pool_frames(frames, method='median')
    except ValueError as exc:
        assert 'unsupported pooling method' in str(exc)
        return
    raise AssertionError('expected ValueError for unsupported pooling')


def test_pool_frames_center():
    frames = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])
    pooled = util_math.pool_frames(frames, method='center')
    assert np.allclose(pooled, [5.0, 5.0])


def test_validate_frames_rejects_one_dimensional_input():
    frames = np.array([1.0, 2.0])
    try:
        util_math.validate_frames(frames)
    except ValueError as exc:
        assert '2d array' in str(exc)
        return
    raise AssertionError('expected ValueError for one-dimensional frames')


def test_validate_frames_rejects_empty_input():
    frames = np.empty((0, 2))
    try:
        util_math.validate_frames(frames)
    except ValueError as exc:
        assert 'must not be empty' in str(exc)
        return
    raise AssertionError('expected ValueError for empty frames')


def test_l2_normalize_accepts_list_input():
    normalized = util_math.l2_normalize([3.0, 4.0])
    assert np.allclose(normalized, [0.6, 0.8])


def test_l2_normalize_preserves_zero_vector():
    normalized = util_math.l2_normalize([0.0, 0.0])
    assert np.allclose(normalized, [0.0, 0.0])


def test_l2_normalize_rows_preserves_zero_rows():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = util_math.l2_normalize_rows(vectors)
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.0, 0.0])
