import numpy as np

from speech_vector_search import store_ingest


def test_build_prototype_artifacts_pools_occurrences():
    occurrences = [
        {
            "label": "hello",
            "unit_type": "word",
            "phraser_key": "p0",
            "echoframe_key": "e0",
            "frames": [[1.0, 3.0], [3.0, 5.0]],
        },
        {
            "label": "bye",
            "unit_type": "phrase",
            "phraser_key": "p1",
            "echoframe_key": "e1",
            "frames": [[0.0, 2.0], [2.0, 4.0]],
        },
    ]
    vectors, metadata, config = store_ingest.build_prototype_artifacts(
        occurrences)
    assert np.allclose(vectors, [[2.0, 4.0], [1.0, 3.0]])
    assert metadata[0]["source_phraser_keys"] == ["p0"]
    assert metadata[0]["source_echoframe_keys"] == ["e0"]
    assert metadata[1]["unit_type"] == "phrase"
    assert config["prototype_method"] == "pooled_occurrence"
    assert config["pooling_method"] == "mean"


def test_load_source_occurrences_uses_store_loader():
    class DummyStore:
        def load_occurrences(self, unit_type=None):
            rows = [
                {"label": "a", "unit_type": "word"},
                {"label": "b", "unit_type": "phone"},
            ]
            if unit_type is None:
                return rows
            return [row for row in rows if row["unit_type"] == unit_type]

    rows = store_ingest.load_source_occurrences(DummyStore(),
        unit_type="word")
    assert rows == [{"label": "a", "unit_type": "word"}]
