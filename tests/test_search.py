import numpy as np

from speech_vector_search.search import PrototypeIndex


def test_brute_force_search_returns_expected_neighbours():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )
    metadata = [
        {"word": "a", "subset_id": 0},
        {"word": "a", "subset_id": 1},
        {"word": "b", "subset_id": 0},
    ]
    index = PrototypeIndex(vectors, metadata, backend="brute_force")
    result = index.query_by_index(0, top_k=2)
    assert result["indices"].tolist() == [0, 1]
    assert result["metadata"][1]["word"] == "a"
