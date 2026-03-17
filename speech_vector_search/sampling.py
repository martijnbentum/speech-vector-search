import numpy as np

from speech_vector_search import utils


def group_token_indices(metadata):
    '''group token indices by word.
    metadata                 token metadata rows
    '''
    groups = {}
    for index, row in enumerate(metadata):
        key = utils.infer_label_key(row)
        word = row[key]
        groups.setdefault(word, []).append(index)
    return groups


def filter_groups_by_count(groups, min_count):
    '''keep groups with enough tokens.
    groups                   word to indices map
    min_count                minimum number of tokens per word
    '''
    selected = {}
    for word, indices in groups.items():
        if len(indices) >= min_count: selected[word] = list(indices)
    return selected


def sample_subsets(indices, subset_size, n_subsets, seed):
    '''sample deterministic non-overlapping subsets.
    indices                  token indices for one word
    subset_size              number of tokens per subset
    n_subsets                number of subsets to sample
    seed                     random seed for shuffling
    '''
    rng = np.random.RandomState(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    subsets = []
    for subset_id in range(n_subsets):
        start = subset_id * subset_size
        end = start + subset_size
        subset = shuffled[start:end]
        if len(subset) < subset_size: break
        subsets.append(subset)
    return subsets


def sample_word_subsets(groups, subset_size, n_subsets, seed=0,
    strict_non_overlapping=True):
    '''sample subsets for all words.
    groups                   word to indices map
    subset_size              number of tokens per subset
    n_subsets                number of subsets per word
    seed                     random seed offset for sampling
    strict_non_overlapping   require full non-overlapping subsets
    '''
    sampled = {}
    for offset, word in enumerate(sorted(groups)):
        indices = groups[word]
        needed = subset_size * n_subsets
        if strict_non_overlapping and len(indices) < needed: continue
        subsets = sample_subsets(indices, subset_size, n_subsets, seed + offset)
        if strict_non_overlapping and len(subsets) != n_subsets: continue
        if subsets: sampled[word] = subsets
    return sampled
