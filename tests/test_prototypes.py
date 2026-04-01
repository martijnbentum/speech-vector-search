import numpy as np

from speech_vector_search import normalize
from speech_vector_search import prototypes


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
        {"label": "a", "phraser_key": "p0", "echoframe_key": "e0",
            "unit_type": "word"},
        {"label": "a", "phraser_key": "p1", "echoframe_key": "e1",
            "unit_type": "word"},
        {"label": "b", "phraser_key": "p2", "echoframe_key": "e2",
            "unit_type": "word"},
        {"label": "b", "phraser_key": "p3", "echoframe_key": "e3",
            "unit_type": "word"},
    ]
    vectors, rows, config = prototypes.build_subset_mean_prototypes(
        embeddings, metadata, subset_size=2, n_subsets=1, seed=0)
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], [1.0, 0.0])
    assert rows[0]["label"] == "a"
    assert rows[0]["unit_type"] == "word"
    assert rows[0]["source_phraser_keys"] == ["p1", "p0"]
    assert rows[0]["source_echoframe_keys"] == ["e1", "e0"]
    assert rows[0]["n_occurrences"] == 2
    assert config["prototype_method"] == "subset_mean"
    assert config["strict_non_overlapping"] is True


def test_l2_normalize():
    vector = np.array([3.0, 4.0])
    normalized = normalize.l2_normalize(vector)
    assert np.allclose(normalized, [0.6, 0.8])
