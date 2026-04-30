import numpy as np


def pool_frames(frames, method='mean'):
    '''reduce frame embeddings to one occurrence vector.
    frames                   frame embedding matrix
    method                   reduction method
    '''
    frames = np.asarray(frames, dtype=float)
    validate_frames(frames)
    if method == 'mean':
        return frames.mean(axis=0)
    if method == 'center':
        n_frames = frames.shape[0]
        center_index = n_frames // 2
        return frames[center_index]
    raise ValueError(f'unsupported pooling method: {method}')


def validate_frames(frames):
    '''validate a frame embedding matrix.'''
    if not isinstance(frames, np.ndarray):
        raise ValueError('frames must be a numpy array')
    if frames.ndim != 2:
        raise ValueError('frames must be a 2d array')
    n_frames = frames.shape[0]
    if n_frames == 0:
        raise ValueError('frames must not be empty')


def l2_normalize(vector):
    '''normalize one vector.
    vector                  input vector
    '''
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.copy()
    return vector / norm


def l2_normalize_rows(vectors):
    '''normalize matrix rows.
    vectors                 input matrix
    '''
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
