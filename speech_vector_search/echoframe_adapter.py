import numpy as np

from speech_vector_search import phraser_adapter
from speech_vector_search import prototypes


try:
    from echoframe import TokenEmbeddings
except ModuleNotFoundError:
    TokenEmbeddings = None


def build_mean_prototype_from_token_embeddings(token_embeddings, label,
    unit_type, subset_id=None):
    '''build one prototype from TokenEmbeddings.
    token_embeddings         echoframe token embeddings
    label                    prototype label
    unit_type                prototype unit type
    subset_id                optional subset number for this prototype
    '''
    token_vectors = _extract_token_vectors(token_embeddings)
    token_rows = _build_token_rows(token_embeddings, label, unit_type,
        token_vectors=token_vectors)
    return prototypes.build_mean_prototype(token_vectors, token_rows,
        subset_id=subset_id, label=label)


def build_subset_prototypes_from_token_embeddings(token_embeddings, label,
    unit_type, subset_size, n_subsets, seed=0, strict_non_overlapping=True):
    '''build subset prototypes from TokenEmbeddings.
    token_embeddings         echoframe token embeddings
    label                    prototype label
    unit_type                prototype unit type
    subset_size              number of tokens per subset
    n_subsets                number of subsets to sample
    seed                     random seed for subset sampling
    strict_non_overlapping   require full non-overlapping subsets
    '''
    token_vectors = _extract_token_vectors(token_embeddings)
    token_rows = _build_token_rows(token_embeddings, label, unit_type,
        token_vectors=token_vectors)
    return prototypes.build_subset_prototypes(label, token_vectors,
        token_rows, subset_size, n_subsets, seed=seed,
        strict_non_overlapping=strict_non_overlapping)


def _build_token_rows(token_embeddings, label, unit_type, token_vectors=None):
    '''build source rows for one TokenEmbeddings input.'''
    _validate_label(label)
    unit_type = phraser_adapter.resolve_unit_type(unit_type)
    echoframe_keys = list(_get_echoframe_keys(token_embeddings))
    if token_vectors is None:
        token_vectors = _extract_token_vectors(token_embeddings)
    if len(echoframe_keys) != len(token_vectors):
        raise ValueError("number of token vectors and echoframe keys must match")
    return [
        {
            "label": label,
            "unit_type": unit_type,
            "echoframe_key": echoframe_key,
        }
        for echoframe_key in echoframe_keys
    ]


def _extract_token_vectors(token_embeddings):
    '''return one stacked array of token vectors.'''
    _validate_token_embeddings(token_embeddings)
    dims = tuple(getattr(token_embeddings, "dims", ()))
    if "frames" in dims:
        raise ValueError("token_embeddings must not contain frames")
    token_vectors = token_embeddings.to_numpy()
    token_vectors = np.asarray(token_vectors)
    if token_vectors.dtype == object:
        raise ValueError("token arrays must share one shape")
    if token_vectors.ndim == 1:
        token_vectors = token_vectors[np.newaxis, :]
    elif token_vectors.ndim == 3:
        if token_vectors.shape[1] != 1:
            raise ValueError("token_embeddings must contain at most one layer")
        token_vectors = token_vectors[:, 0, :]
    elif token_vectors.ndim != 2:
        raise ValueError("token_embeddings must stack into a 2D array")
    return token_vectors


def _get_echoframe_keys(token_embeddings):
    '''return token-level echoframe keys.'''
    if not hasattr(token_embeddings, "echoframe_keys"):
        raise ValueError("token_embeddings must expose echoframe_keys")
    return token_embeddings.echoframe_keys


def _validate_label(label):
    '''validate one explicit prototype label.'''
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")


def _validate_token_embeddings(token_embeddings):
    '''validate the adapter input type.'''
    if TokenEmbeddings is not None:
        if isinstance(token_embeddings, TokenEmbeddings):
            return
        raise ValueError("token_embeddings must be an echoframe.TokenEmbeddings")
    if token_embeddings.__class__.__name__ != "TokenEmbeddings":
        raise ValueError("token_embeddings must be an echoframe.TokenEmbeddings")
