import tempfile

import numpy as np

from speech_vector_search import io
from speech_vector_search import locations


def test_save_and_load_prototypes_with_name():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [
        {
            "label": "a",
            "unit_type": "word",
            "source_phraser_keys": ["p0"],
            "source_echoframe_keys": ["e0"],
            "n_occurrences": 1,
        },
        {
            "label": "b",
            "unit_type": "word",
            "source_phraser_keys": ["p1"],
            "source_echoframe_keys": ["e1"],
            "n_occurrences": 1,
        },
    ]
    with tempfile.TemporaryDirectory() as directory:
        paths = io.save_prototypes(vectors, metadata, directory=directory,
            name="prototypes")
        loaded_vectors, loaded_metadata = io.load_prototypes(
            name="prototypes",
            directory=directory,
        )
    assert paths["vectors"] == locations.make_path(directory, "prototypes",
        "vectors", overwrite=True)
    assert paths["metadata"] == locations.make_path(directory, "prototypes",
        "metadata", overwrite=True)
    assert np.allclose(loaded_vectors, vectors)
    assert loaded_metadata == metadata


def test_save_and_load_prototypes_with_default_name():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [
        {
            "label": "a",
            "unit_type": "word",
            "source_phraser_keys": ["p0"],
            "source_echoframe_keys": ["e0"],
            "n_occurrences": 1,
        },
        {
            "label": "b",
            "unit_type": "word",
            "source_phraser_keys": ["p1"],
            "source_echoframe_keys": ["e1"],
            "n_occurrences": 1,
        },
    ]
    with tempfile.TemporaryDirectory() as directory:
        paths = io.save_prototypes(vectors, metadata, directory=directory)
        loaded_vectors, loaded_metadata = io.load_prototypes(
            directory=directory)
    assert paths["vectors"] == locations.make_path(directory, "prototypes",
        "vectors", overwrite=True)
    assert paths["metadata"] == locations.make_path(directory, "prototypes",
        "metadata", overwrite=True)
    assert np.allclose(loaded_vectors, vectors)
    assert loaded_metadata == metadata


def test_save_prototypes_raises_when_files_exist():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [
        {
            "label": "a",
            "unit_type": "word",
            "source_phraser_keys": ["p0"],
            "source_echoframe_keys": ["e0"],
            "n_occurrences": 1,
        },
        {
            "label": "b",
            "unit_type": "word",
            "source_phraser_keys": ["p1"],
            "source_echoframe_keys": ["e1"],
            "n_occurrences": 1,
        },
    ]
    with tempfile.TemporaryDirectory() as directory:
        io.save_prototypes(vectors, metadata, directory=directory)
        try:
            io.save_prototypes(vectors, metadata, directory=directory)
        except FileExistsError:
            return
    raise AssertionError("expected FileExistsError when prototype files exist")


def test_save_prototypes_ignores_existing_config_when_config_is_none():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [
        {
            "label": "a",
            "unit_type": "word",
            "source_phraser_keys": ["p0"],
            "source_echoframe_keys": ["e0"],
            "n_occurrences": 1,
        },
        {
            "label": "b",
            "unit_type": "word",
            "source_phraser_keys": ["p1"],
            "source_echoframe_keys": ["e1"],
            "n_occurrences": 1,
        },
    ]
    with tempfile.TemporaryDirectory() as directory:
        config_path = locations.make_path(directory, "demo", "config",
            overwrite=True)
        config_path.write_text("{}")
        paths = io.save_prototypes(vectors, metadata, directory=directory,
            name="demo")
    assert paths["config"] is None


def test_save_prototypes_creates_nested_directories():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [
        {
            "label": "a",
            "unit_type": "word",
            "source_phraser_keys": ["p0"],
            "source_echoframe_keys": ["e0"],
            "n_occurrences": 1,
        },
        {
            "label": "b",
            "unit_type": "word",
            "source_phraser_keys": ["p1"],
            "source_echoframe_keys": ["e1"],
            "n_occurrences": 1,
        },
    ]
    with tempfile.TemporaryDirectory() as directory:
        nested_directory = (locations.Path(directory) / "runs" / "2026-04-07"
            / "demo")
        paths = io.save_prototypes(vectors, metadata, directory=nested_directory,
            name="nested")
        assert paths["vectors"].exists()
        assert paths["metadata"].exists()


def test_prototype_paths_use_directory_layout():
    with tempfile.TemporaryDirectory() as directory:
        name = "word_wav2vec2_layer07_mean"
        vectors_path = locations.make_path(directory, name, "vectors",
            overwrite=True)
        metadata_path = locations.make_path(directory, name, "metadata",
            overwrite=True)
        config_path = locations.make_path(directory, name, "config",
            overwrite=True)
    assert vectors_path.name == "word_wav2vec2_layer07_mean_vectors.npy"
    assert metadata_path.name == "word_wav2vec2_layer07_mean_metadata.jsonl"
    assert config_path.name == "word_wav2vec2_layer07_mean_config.json"
