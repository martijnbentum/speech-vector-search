import numpy as np

from speech_vector_search import metadata
from speech_vector_search import normalize
from speech_vector_search import sampling
from speech_vector_search import utils


def build_subset_prototypes(label, embeddings, token_metadata, subset_size,
    n_subsets, seed=0, strict_non_overlapping=True):
    '''build normalized subset-mean prototypes for one label.
    label                    label string
    embeddings               token embedding matrix
    token_metadata           token metadata rows (dicts or PrototypeMetadata)
    subset_size              number of tokens per subset
    n_subsets                number of subsets to sample
    seed                     random seed
    strict_non_overlapping   require full non-overlapping subsets
    '''
    _validate_aligned(embeddings, token_metadata)
    label = _resolve_single_label(token_metadata, label)
    subsets = sampling.sample_subsets(range(len(token_metadata)), subset_size,
        n_subsets, seed)
    if strict_non_overlapping and len(subsets) != n_subsets:
        raise ValueError(
            f"not enough tokens for {n_subsets} non-overlapping subsets")
    vectors, rows = [], []
    for subset_index, subset_indices in enumerate(subsets):
        subset_embeddings = embeddings[subset_indices]
        subset_rows = [token_metadata[i] for i in subset_indices]
        vector, row = build_mean_prototype(subset_embeddings, subset_rows,
            subset_index, label)
        vectors.append(vector)
        rows.append(row)
    if vectors:
        vectors = np.vstack(vectors)
    else:
        vectors = np.zeros((0, embeddings.shape[1]), dtype=float)
    required_count = subset_size * n_subsets if strict_non_overlapping else subset_size
    config = {
        "prototype_method": "subset_mean",
        "subset_size": subset_size,
        "n_subsets": n_subsets,
        "min_count": required_count,
        "seed": seed,
        "strict_non_overlapping": strict_non_overlapping,
        "label": label,
    }
    return vectors, rows, config


def build_mean_prototype(embeddings, token_metadata, subset_id=None,
    label=None):
    '''build one normalized mean prototype from aligned token embeddings.
    embeddings               token embedding matrix
    token_metadata           token metadata rows
    subset_id                optional subset index
    label                    optional label to require
    '''
    embeddings = np.asarray(embeddings, dtype=float)
    _validate_aligned(embeddings, token_metadata)
    label = _resolve_single_label(token_metadata, label)
    vector = normalize.l2_normalize(embeddings.mean(axis=0))
    unit_type = metadata.resolve_shared_unit_type(token_metadata)
    source_keys = _collect_source_keys(token_metadata)
    row = metadata.PrototypeMetadata(label, unit_type, source_keys,
        subset_id=subset_id)
    return vector, row


def validate_rows(metadata_items):
    '''validate a list of PrototypeMetadata items.'''
    metadata.validate_rows(metadata_items)


def _validate_aligned(embeddings, rows):
    if len(embeddings) != len(rows):
        raise ValueError("number of embeddings and metadata rows must match")


def _resolve_single_label(rows, label=None):
    '''verify all rows share one label and return it.'''
    labels = []
    for row in rows:
        v = utils.label_value(row)
        if v not in labels:
            labels.append(v)
    if not labels:
        raise ValueError("metadata must not be empty")
    if len(labels) != 1:
        raise ValueError("metadata must contain exactly one label")
    if label is not None and label != labels[0]:
        raise ValueError(f"metadata label mismatch: expected {label!r}")
    return labels[0]


def _collect_source_keys(rows):
    '''collect source identifier strings from token metadata rows.'''
    keys = []
    for row in rows:
        if isinstance(row, metadata.PrototypeMetadata):
            keys.extend(row.source_echoframe_keys)
        elif isinstance(row, dict):
            if "echoframe_key" in row:
                keys.append(row["echoframe_key"])
            elif "source_echoframe_keys" in row:
                keys.extend(row["source_echoframe_keys"])
            elif "id" in row:
                keys.append(row["id"])
            else:
                raise ValueError(
                    "token row must contain 'echoframe_key', "
                    "'source_echoframe_keys', or 'id'")
        else:
            raise ValueError("token row must be a dict or PrototypeMetadata")
    if not keys:
        raise ValueError("no source keys found in token rows")
    return keys
