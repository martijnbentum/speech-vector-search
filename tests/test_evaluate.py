import numpy as np

from speech_vector_search.evaluate import evaluate_same_word_retrieval


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
        {"word": "a", "subset_id": 0},
        {"word": "a", "subset_id": 1},
        {"word": "b", "subset_id": 0},
        {"word": "b", "subset_id": 1},
    ]
    result = evaluate_same_word_retrieval(vectors, metadata, top_k=1)
    assert result["n_words"] == 2
    assert result["n_prototypes"] == 4
    assert result["top_k_same_word"] == 1.0
    assert result["mean_same_word_rank"] >= 2.0
