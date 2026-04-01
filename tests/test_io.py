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
            directory=directory,
            name="prototypes",
        )
    assert paths["vectors"] == locations.prototype_vectors_path(directory,
        "prototypes")
    assert paths["metadata"] == locations.prototype_metadata_path(directory,
        "prototypes")
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


def test_prototype_paths_use_directory_layout():
    with tempfile.TemporaryDirectory() as directory:
        artifact_directory = locations.prototype_directory(directory,
            "word_wav2vec2_layer07_mean")
        vectors_path = locations.prototype_vectors_path(directory,
            "word_wav2vec2_layer07_mean")
        metadata_path = locations.prototype_metadata_path(directory,
            "word_wav2vec2_layer07_mean")
        config_path = locations.config_path(directory,
            "word_wav2vec2_layer07_mean")
    assert vectors_path == artifact_directory / "prototypes.npy"
    assert metadata_path == artifact_directory / "metadata.jsonl"
    assert config_path == artifact_directory / "config.json"
