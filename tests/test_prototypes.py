import numpy as np

from speech_vector_search.normalize import l2_normalize
from speech_vector_search.prototypes import build_subset_mean_prototypes


def test_prototype_mean_and_metadata():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    metadata = [
        {"word": "a", "id": "0", "speaker": "s1"},
        {"word": "a", "id": "1", "speaker": "s1"},
        {"word": "b", "id": "2", "speaker": "s2"},
        {"word": "b", "id": "3", "speaker": "s3"},
    ]
    vectors, rows, config = build_subset_mean_prototypes(
        embeddings,
        metadata,
        subset_size=2,
        n_subsets=1,
        seed=0,
    )
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], [1.0, 0.0])
    assert rows[0]["word"] == "a"
    assert rows[0]["source_token_indices"] == [1, 0]
    assert rows[0]["speaker_summary"] == {"s1": 2}
    assert config["strict_non_overlapping"] is True


def test_l2_normalize():
    vector = np.array([3.0, 4.0])
    normalized = l2_normalize(vector)
    assert np.allclose(normalized, [0.6, 0.8])
