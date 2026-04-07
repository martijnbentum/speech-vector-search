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


def test_extract_echoframe_metadata_uses_label_tag():
    class DummyMetadata:
        def __init__(self, phraser_key, entry_id):
            self.phraser_key = phraser_key
            self.entry_id = entry_id

    class DummyStore:
        def find_by_tag(self, tag):
            assert tag == "hello"
            return [
                DummyMetadata("p0", "e0"),
                DummyMetadata("p1", "e1"),
            ]

    rows = store_ingest.extract_echoframe_metadata(DummyStore(), "hello",
        unit_type="phrase")
    assert [row["label"] for row in rows] == ["hello", "hello"]
    assert [row["unit_type"] for row in rows] == ["phrase", "phrase"]
    assert [row["phraser_key"] for row in rows] == ["p0", "p1"]
    assert [row["echoframe_key"] for row in rows] == ["e0", "e1"]
    assert rows[0]["echoframe_metadata"].entry_id == "e0"


def test_extract_echoframe_metadata_uses_metadata_unit_type():
    class DummyMetadata:
        def __init__(self, phraser_key, entry_id, unit_type):
            self.phraser_key = phraser_key
            self.entry_id = entry_id
            self.unit_type = unit_type

    class DummyStore:
        def find_by_tag(self, tag):
            assert tag == 'hello'
            return [DummyMetadata('p0', 'e0', 'phrase')]

    rows = store_ingest.extract_echoframe_metadata(DummyStore(), 'hello')
    assert rows[0]['unit_type'] == 'phrase'


def test_extract_echoframe_metadata_requires_unit_type_when_missing():
    class DummyMetadata:
        def __init__(self, phraser_key, entry_id):
            self.phraser_key = phraser_key
            self.entry_id = entry_id

    class DummyStore:
        def find_by_tag(self, tag):
            assert tag == 'hello'
            return [DummyMetadata('p0', 'e0')]

    try:
        store_ingest.extract_echoframe_metadata(DummyStore(), 'hello')
    except ValueError as exc:
        assert 'unit_type is required' in str(exc)
        return
    raise AssertionError('expected ValueError for missing unit_type')


def test_load_echoframe_payloads_loads_frames_for_each_row():
    class DummyMetadata:
        def __init__(self, entry_id):
            self.entry_id = entry_id

    class DummyStorage:
        def load(self, metadata):
            return {
                "e0": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "e1": np.array([[5.0, 6.0]]),
            }[metadata.entry_id]

    class DummyStore:
        storage = DummyStorage()

    rows = [
        {
            "label": "hello",
            "unit_type": "word",
            "phraser_key": "p0",
            "echoframe_key": "e0",
            "echoframe_metadata": DummyMetadata("e0"),
        },
        {
            "label": "bye",
            "unit_type": "word",
            "phraser_key": "p1",
            "echoframe_key": "e1",
            "echoframe_metadata": DummyMetadata("e1"),
        },
    ]

    loaded = store_ingest.load_echoframe_payloads(DummyStore(), rows)

    assert np.allclose(loaded[0]["frames"], [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(loaded[1]["frames"], [[5.0, 6.0]])
    assert loaded[0]["echoframe_key"] == "e0"
