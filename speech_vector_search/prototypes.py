import numpy as np

from speech_vector_search import normalize
from speech_vector_search import phraser_adapter
from speech_vector_search import prototype_metadata
from speech_vector_search import sampling
from speech_vector_search import utils


def build_subset_prototypes(label, embeddings, metadata, subset_size, n_subsets,
    seed=0, strict_non_overlapping=True):
    '''build normalized subset-mean prototypes for one label only.
    label                    label
    embeddings               token embeddings for one label
    metadata                 token metadata rows for one label
    subset_size              number of tokens per subset
    n_subsets                number of subsets to sample
    seed                     random seed for subset sampling
    strict_non_overlapping   require full non-overlapping subsets
    '''
    validate_embedding_metadata(embeddings, metadata)
    label = resolve_single_label(metadata, label=label)
    if strict_non_overlapping: required_count = subset_size * n_subsets
    else: required_count = subset_size
    subsets = sampling.sample_subsets(range(len(metadata)), subset_size,
        n_subsets, seed)
    if strict_non_overlapping and len(subsets) != n_subsets:
        m = f"not enough tokens for {n_subsets} non-overlapping subsets"
        raise ValueError(m)
    vectors, rows = [], []
    for subset_index, subset_indices in enumerate(subsets):
        subset_embeddings = embeddings[subset_indices]
        token_rows = [metadata[index] for index in subset_indices]
        vector, row = build_mean_prototype(subset_embeddings, token_rows,
            subset_index, label)
        vectors.append(vector)
        rows.append(row)
    if vectors: vectors = np.vstack(vectors)
    else: vectors = np.zeros((0, embeddings.shape[1]), dtype=float)
    config = make_config(subset_size, n_subsets, required_count, seed,
        strict_non_overlapping, label=label)
    return vectors, rows, config


def build_mean_prototype(embeddings, metadata, subset_id=None, label=None):
    '''build one normalized mean prototype from aligned source tokens.
    embeddings               token embeddings for one label
    metadata                 token metadata rows for one label
    label                    optional label to require
    '''
    embeddings = np.asarray(embeddings, dtype=float)
    validate_embedding_metadata(embeddings, metadata)
    label = resolve_single_label(metadata, label=label)
    mean_vector = embeddings.mean(axis=0)
    vector = normalize.l2_normalize(mean_vector)
    row = make_prototype_row(label, subset_id, metadata)
    return vector, row

def make_config(subset_size, n_subsets, min_count, seed,
    strict_non_overlapping, label=None):
    '''build config for subset-mean prototype generation.
    subset_size              number of tokens per subset
    n_subsets                number of subsets to sample
    min_count                minimum number of tokens required
    seed                     random seed for subset sampling
    strict_non_overlapping   require full non-overlapping subsets
    label                    optional single label for the input
    '''
    config = {
        "prototype_method": "subset_mean",
        "subset_size": subset_size,
        "n_subsets": n_subsets,
        "min_count": min_count,
        "seed": seed,
        "strict_non_overlapping": strict_non_overlapping,
    }
    if label is not None: config["label"] = label
    return config


def make_prototype_row(word, subset_id, token_rows):
    '''build metadata row for one prototype.
    word                     prototype label
    subset_id                subset number for this prototype
    token_rows               metadata rows for subset tokens
    '''
    unit_type = infer_unit_type(token_rows)
    source_phraser_keys = gather_source_keys(token_rows, "phraser_key",
        "source_phraser_keys")
    source_echoframe_keys = gather_source_keys(token_rows, "echoframe_key",
        "source_echoframe_keys")
    return prototype_metadata.make_prototype_row(word, unit_type,
        source_phraser_keys, source_echoframe_keys, subset_id=subset_id)


def validate_row(row):
    '''validate one prototype metadata row.
    row                      prototype metadata record
    '''
    prototype_metadata.validate_row(row)


def validate_rows(rows):
    '''validate stored prototype metadata rows.
    rows                     prototype metadata records
    '''
    prototype_metadata.validate_rows(rows)


def infer_unit_type(rows):
    '''infer one shared unit type from source rows.
    rows                     metadata rows in subset
    '''
    return phraser_adapter.resolve_shared_unit_type(rows)


def validate_embedding_metadata(embeddings, metadata):
    '''validate aligned embedding and metadata inputs.
    embeddings               token embeddings
    metadata                 token metadata rows
    '''
    if len(embeddings) != len(metadata):
        raise ValueError("number of embeddings and metadata rows must match")


def resolve_single_label(metadata, label=None):
    '''resolve and validate that metadata belongs to one label.
    metadata                 token metadata rows
    label                    optional label to require
    '''
    labels = []
    for row in metadata:
        row_label = utils.label_value(row)
        if row_label not in labels: labels.append(row_label)
    if not labels: raise ValueError("metadata must not be empty")
    if len(labels) != 1: raise ValueError("metadata must contain exactly one label")
    if label is not None and label != labels[0]:
        raise ValueError(f"metadata label mismatch: expected '{label}'")
    return labels[0]


def gather_source_keys(rows, singular_key, plural_key):
    '''collect source identifiers from source or prototype rows.
    rows                     metadata rows in subset
    singular_key             single-source field name
    plural_key               multi-source field name
    '''
    values = []
    for row in rows:
        if plural_key in row:
            items = row[plural_key]
        elif singular_key in row:
            items = [row[singular_key]]
        elif "id" in row:
            items = [row["id"]]
        else:
            items = [fallback_source_key(row, singular_key)]
        for item in items:
            if not item:
                raise ValueError(f"metadata row contains empty '{singular_key}'")
            values.append(item)
    return values


def fallback_source_key(row, singular_key):
    '''build a deterministic fallback source key for generic token rows.
    row                     metadata row
    singular_key            requested single-source field name
    '''
    label = utils.label_value(row)
    suffix = singular_key.replace("_key", "")
    return f"{label}:{suffix}"
