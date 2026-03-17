from collections import Counter

import numpy as np

from speech_vector_search.normalize import l2_normalize
from speech_vector_search.sampling import filter_groups_by_count, group_token_indices, sample_word_subsets


def summarize_speakers(rows):
    '''count speakers in subset rows.
    rows                     metadata rows in subset
    '''
    speakers = [row.get("speaker") for row in rows if row.get("speaker") is not None]
    if not speakers:
        return None
    counts = Counter(speakers)
    return dict(sorted(counts.items()))


def make_prototype_row(word, subset_id, subset_indices, token_rows):
    '''build metadata row for one prototype.
    word                     word label
    '''
    row = {
        "word": word,
        "subset_id": subset_id,
        "n_tokens": len(subset_indices),
        "source_token_indices": list(subset_indices),
    }
    speaker_summary = summarize_speakers(token_rows)
    if speaker_summary is not None:
        row["speaker_summary"] = speaker_summary
    return row


def build_subset_mean_prototypes(
    embeddings,
    metadata,
    subset_size,
    n_subsets,
    min_count=None,
    seed=0,
    strict_non_overlapping=True,
):
    '''build normalized subset-mean prototypes.
    embeddings               token embeddings
    '''
    embeddings = np.asarray(embeddings, dtype=float)
    groups = group_token_indices(metadata)
    required = subset_size * n_subsets if strict_non_overlapping else subset_size
    if min_count is None:
        min_count = required
    min_count = max(min_count, required) if strict_non_overlapping else min_count
    groups = filter_groups_by_count(groups, min_count)
    sampled = sample_word_subsets(
        groups,
        subset_size=subset_size,
        n_subsets=n_subsets,
        seed=seed,
        strict_non_overlapping=strict_non_overlapping,
    )

    vectors = []
    rows = []
    for word in sorted(sampled):
        for subset_id, subset_indices in enumerate(sampled[word]):
            subset_vectors = embeddings[subset_indices]
            mean_vector = subset_vectors.mean(axis=0)
            vectors.append(l2_normalize(mean_vector))
            token_rows = [metadata[index] for index in subset_indices]
            rows.append(make_prototype_row(word, subset_id, subset_indices, token_rows))

    if vectors:
        vectors = np.vstack(vectors)
    else:
        vectors = np.zeros((0, embeddings.shape[1]), dtype=float)

    config = {
        "subset_size": subset_size,
        "n_subsets": n_subsets,
        "min_count": min_count,
        "seed": seed,
        "strict_non_overlapping": strict_non_overlapping,
    }
    return vectors, rows, config
