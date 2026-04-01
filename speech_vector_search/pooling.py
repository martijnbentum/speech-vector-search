import numpy as np


def pool_frames(frames, method="mean"):
    '''reduce frame embeddings to one occurrence vector.
    frames                   frame embedding matrix
    method                   reduction method
    '''
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 2:
        raise ValueError("frames must be a 2d array")
    if len(frames) == 0:
        raise ValueError("frames must not be empty")
    if method == "mean":
        return frames.mean(axis=0)
    if method == "max":
        return frames.max(axis=0)
    if method == "first":
        return frames[0]
    raise ValueError(f"unsupported pooling method: {method}")
