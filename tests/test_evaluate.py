import numpy as np

from speech_vector_search import evaluate


def test_evaluation_metrics_on_tiny_dataset():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ]
    )
    metadata = [
        {"label": "a", "subset_id": 0},
        {"label": "a", "subset_id": 1},
        {"label": "b", "subset_id": 0},
        {"label": "b", "subset_id": 1},
    ]
    result = evaluate.evaluate_same_word_retrieval(vectors, metadata, top_k=1)
    assert result["n_words"] == 2
    assert result["n_prototypes"] == 4
    assert result["top_k_same_word"] == 1.0
    assert result["mean_same_word_rank"] >= 2.0


def test_evaluation_metrics_accept_label_rows():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ]
    )
    metadata = [
        {"label": "a", "unit_type": "word", "source_echoframe_keys": ["e0"],
            "n_occurrences": 1},
        {"label": "a", "unit_type": "word", "source_echoframe_keys": ["e1"],
            "n_occurrences": 1},
        {"label": "b", "unit_type": "word", "source_echoframe_keys": ["e2"],
            "n_occurrences": 1},
        {"label": "b", "unit_type": "word", "source_echoframe_keys": ["e3"],
            "n_occurrences": 1},
    ]
    result = evaluate.evaluate_same_word_retrieval(vectors, metadata, top_k=1)
    assert result["n_words"] == 2
    assert result["n_prototypes"] == 4
    assert result["top_k_same_word"] == 1.0
