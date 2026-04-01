import numpy as np

from speech_vector_search import search


def test_brute_force_search_returns_expected_neighbours():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )
    metadata = [
        {"label": "a", "subset_id": 0},
        {"label": "a", "subset_id": 1},
        {"label": "b", "subset_id": 0},
    ]
    index = search.PrototypeIndex(vectors, metadata, backend="brute_force")
    result = index.query_by_index(0, top_k=2)
    assert result["indices"].tolist() == [0, 1]
    assert result["metadata"][1]["label"] == "a"
