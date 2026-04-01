import numpy as np

from speech_vector_search import pooling


def test_pool_frames_mean():
    frames = np.array([[1.0, 3.0], [3.0, 5.0]])
    pooled = pooling.pool_frames(frames)
    assert np.allclose(pooled, [2.0, 4.0])


def test_pool_frames_rejects_unknown_method():
    frames = np.array([[1.0, 2.0]])
    try:
        pooling.pool_frames(frames, method="median")
    except ValueError as exc:
        assert "unsupported pooling method" in str(exc)
        return
    raise AssertionError("expected ValueError for unsupported pooling")
