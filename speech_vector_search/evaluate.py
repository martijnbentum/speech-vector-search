from collections import Counter

import numpy as np

from speech_vector_search.search import PrototypeIndex


def _same_word_mask(metadata, query_index):
    '''mark same-word prototypes.
    metadata                 prototype metadata
    '''
    word = metadata[query_index]["word"]
    mask = np.array([row["word"] == word for row in metadata], dtype=bool)
    mask[query_index] = False
    return mask


def top_k_same_word_retrieval(vectors, metadata, top_k=5):
    '''compute top-k same-word hit rate.
    vectors                  prototype matrix
    '''
    index = PrototypeIndex(vectors, metadata, backend="brute_force")
    hits = []
    for query_index in range(len(metadata)):
        result = index.query_by_index(query_index, top_k=min(top_k + 1, len(metadata)))
        found = False
        for neighbour in result["indices"]:
            if neighbour == query_index:
                continue
            if metadata[int(neighbour)]["word"] == metadata[query_index]["word"]:
                found = True
                break
        hits.append(found)
    return float(np.mean(hits)) if hits else 0.0


def mean_same_word_rank(vectors, metadata):
    '''compute mean rank of first same-word match.
    vectors                  prototype matrix
    '''
    index = PrototypeIndex(vectors, metadata, backend="brute_force")
    ranks = []
    for query_index in range(len(metadata)):
        result = index.query_by_index(query_index, top_k=len(metadata))
        rank = len(metadata)
        for position, neighbour in enumerate(result["indices"], start=1):
            if neighbour == query_index:
                continue
            if metadata[int(neighbour)]["word"] == metadata[query_index]["word"]:
                rank = position
                break
        ranks.append(rank)
    return float(np.mean(ranks)) if ranks else 0.0


def average_within_word_similarity(vectors, metadata):
    '''compute average within-word cosine similarity.
    vectors                  prototype matrix
    '''
    vectors = np.asarray(vectors, dtype=float)
    similarities = []
    for word in sorted({row["word"] for row in metadata}):
        indices = [i for i, row in enumerate(metadata) if row["word"] == word]
        if len(indices) < 2:
            continue
        subset = vectors[indices]
        scores = np.dot(subset, subset.T)
        upper = np.triu_indices(len(indices), k=1)
        similarities.extend(scores[upper].tolist())
    return float(np.mean(similarities)) if similarities else 0.0


def summarize_prototypes(vectors, metadata):
    '''summarize prototype collection.
    vectors                  prototype matrix
    '''
    counts = Counter(row["word"] for row in metadata)
    return {
        "n_words": len(counts),
        "n_prototypes": len(metadata),
        "prototypes_per_word": dict(sorted(counts.items())),
        "average_within_word_cosine": average_within_word_similarity(vectors, metadata),
    }


def evaluate_same_word_retrieval(vectors, metadata, top_k=5):
    '''run simple same-word retrieval evaluation.
    vectors                  prototype matrix
    '''
    summary = summarize_prototypes(vectors, metadata)
    summary["top_k_same_word"] = top_k_same_word_retrieval(vectors, metadata, top_k=top_k)
    summary["mean_same_word_rank"] = mean_same_word_rank(vectors, metadata)
    return summary
