from collections import Counter

import numpy as np

from speech_vector_search import search
from speech_vector_search import utils


def top_k_same_word_retrieval(vectors, metadata, top_k=5):
    '''measure how often a same-word prototype appears in the top-k results.
    vectors                  prototype matrix
    metadata                 prototype metadata rows
    top_k                    number of neighbours to inspect
    '''
    index = search.PrototypeIndex(vectors, metadata, backend="brute_force")
    hits = []
    for query_index in range(len(metadata)):
        search_depth = min(top_k + 1, len(metadata))
        result = index.query_by_index(query_index, top_k=search_depth)
        found = False
        for neighbour in result["indices"]:
            neighbour = int(neighbour)
            if neighbour == query_index: continue
            if label_value(metadata[neighbour]) == label_value(
                metadata[query_index]):
                found = True
                break
        hits.append(found)
    return float(np.mean(hits)) if hits else 0.0


def mean_same_word_rank(vectors, metadata):
    '''measure the average rank of the first same-word neighbour.
    vectors                  prototype matrix
    metadata                 prototype metadata rows
    '''
    index = search.PrototypeIndex(vectors, metadata, backend="brute_force")
    ranks = []
    for query_index in range(len(metadata)):
        result = index.query_by_index(query_index, top_k=len(metadata))
        rank = len(metadata)
        for position, neighbour in enumerate(result["indices"], start=1):
            neighbour = int(neighbour)
            if neighbour == query_index: continue
            if label_value(metadata[neighbour]) == label_value(
                metadata[query_index]):
                rank = position
                break
        ranks.append(rank)
    return float(np.mean(ranks)) if ranks else 0.0


def average_within_word_similarity(vectors, metadata):
    '''compute the mean cosine similarity between prototypes of the same word.
    vectors                  prototype matrix
    metadata                 prototype metadata rows
    '''
    vectors = np.asarray(vectors, dtype=float)
    similarities = []
    labels = [label_value(row) for row in metadata]
    for word in sorted(set(labels)):
        indices = [i for i, label in enumerate(labels) if label == word]
        if len(indices) < 2: continue
        subset = vectors[indices]
        scores = np.dot(subset, subset.T)
        upper = np.triu_indices(len(indices), k=1)
        similarities.extend(scores[upper].tolist())
    return float(np.mean(similarities)) if similarities else 0.0


def summarize_prototypes(vectors, metadata):
    '''summarize how many words and prototypes are present in the collection.
    vectors                  prototype matrix
    metadata                 prototype metadata rows
    '''
    counts = Counter(label_value(row) for row in metadata)
    within_word_cosine = average_within_word_similarity(vectors, metadata)
    return {
        "n_words": len(counts),
        "n_prototypes": len(metadata),
        "prototypes_per_word": dict(sorted(counts.items())),
        "average_within_word_cosine": within_word_cosine,
    }


def evaluate_same_word_retrieval(vectors, metadata, top_k=5):
    '''run a small retrieval report over the prototype collection.
    vectors                  prototype matrix
    metadata                 prototype metadata rows
    top_k                    number of neighbours to inspect
    '''
    summary = summarize_prototypes(vectors, metadata)
    top_k_score = top_k_same_word_retrieval(vectors, metadata, top_k=top_k)
    summary["top_k_same_word"] = top_k_score
    summary["mean_same_word_rank"] = mean_same_word_rank(vectors, metadata)
    return summary


def _same_word_mask(metadata, query_index):
    '''mark prototypes that share the query word.
    metadata                 prototype metadata rows
    query_index              prototype index used as query
    '''
    word = label_value(metadata[query_index])
    mask = np.array([label_value(row) == word for row in metadata], dtype=bool)
    mask[query_index] = False
    return mask


def label_value(row):
    '''return the comparable label for one prototype row.
    row                     prototype metadata row
    '''
    return utils.label_value(row)
