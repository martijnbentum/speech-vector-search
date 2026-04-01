from speech_vector_search import sampling


def test_group_token_indices():
    metadata = [
        {"label": "a", "id": "0"},
        {"label": "b", "id": "1"},
        {"label": "a", "id": "2"},
    ]
    groups = sampling.group_token_indices(metadata)
    assert groups == {"a": [0, 2], "b": [1]}


def test_non_overlapping_subsets():
    groups = {"label": list(range(8))}
    sampled = sampling.sample_word_subsets(groups, subset_size=2, n_subsets=3,
        seed=4)
    subsets = sampled["label"]
    used = []
    for subset in subsets:
        used.extend(subset)
    assert len(used) == len(set(used))


def test_sampling_is_deterministic():
    groups = {"label": list(range(10))}
    first = sampling.sample_word_subsets(groups, subset_size=2, n_subsets=3,
        seed=9)
    second = sampling.sample_word_subsets(groups, subset_size=2, n_subsets=3,
        seed=9)
    assert first == second
