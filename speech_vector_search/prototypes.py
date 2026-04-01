import numpy as np

from speech_vector_search import normalize
from speech_vector_search import sampling
from speech_vector_search import utils


def build_subset_mean_prototypes(embeddings, metadata, subset_size, n_subsets,
    min_count=None, seed=0, strict_non_overlapping=True):
    '''build normalized subset-mean prototypes.
    embeddings               token embeddings
    metadata                 token metadata rows
    subset_size              number of tokens per subset
    n_subsets                number of subsets per word
    min_count                minimum tokens required for a word
    seed                     random seed for subset sampling
    strict_non_overlapping   require full non-overlapping subsets
    '''
    embeddings = np.asarray(embeddings, dtype=float)
    groups = sampling.group_token_indices(metadata)
    if strict_non_overlapping: required_count = subset_size * n_subsets
    else: required_count = subset_size
    if min_count is None: min_count = required_count
    if strict_non_overlapping:
        min_count = max(min_count, required_count)
    groups = sampling.filter_groups_by_count(groups, min_count)
    sampled = sampling.sample_word_subsets(groups, subset_size=subset_size,
        n_subsets=n_subsets, seed=seed,
        strict_non_overlapping=strict_non_overlapping)

    vectors = []
    rows = []
    for word in sorted(sampled):
        for subset_id, subset_indices in enumerate(sampled[word]):
            subset_vectors = embeddings[subset_indices]
            mean_vector = subset_vectors.mean(axis=0)
            vectors.append(normalize.l2_normalize(mean_vector))
            token_rows = [metadata[index] for index in subset_indices]
            prototype_row = make_prototype_row(word, subset_id, subset_indices,
                token_rows)
            rows.append(prototype_row)

    if vectors: vectors = np.vstack(vectors)
    else: vectors = np.zeros((0, embeddings.shape[1]), dtype=float)

    config = {
        "prototype_method": "subset_mean",
        "subset_size": subset_size,
        "n_subsets": n_subsets,
        "min_count": min_count,
        "seed": seed,
        "strict_non_overlapping": strict_non_overlapping,
    }
    return vectors, rows, config


def make_prototype_row(word, subset_id, subset_indices, token_rows):
    '''build metadata row for one prototype.
    word                     prototype label
    subset_id                subset number for this prototype
    subset_indices           source token indices in the subset
    token_rows               metadata rows for subset tokens
    '''
    unit_type = infer_unit_type(token_rows)
    source_phraser_keys = gather_source_keys(token_rows, "phraser_key",
        "source_phraser_keys")
    source_echoframe_keys = gather_source_keys(token_rows, "echoframe_key",
        "source_echoframe_keys")
    return {
        "label": word,
        "unit_type": unit_type,
        "source_phraser_keys": source_phraser_keys,
        "source_echoframe_keys": source_echoframe_keys,
        "n_occurrences": len(source_phraser_keys),
        "subset_id": subset_id,
    }


def infer_unit_type(rows):
    '''infer one shared unit type from source rows.
    rows                     metadata rows in subset
    '''
    unit_types = []
    for row in rows:
        unit_type = row.get("unit_type")
        if unit_type is None:
            unit_type = "word"
        unit_types.append(unit_type)
    unique = sorted(set(unit_types))
    if len(unique) != 1:
        raise ValueError("subset rows must share one unit_type")
    return unique[0]


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
