import tempfile

import numpy as np

from speech_vector_search import io
from speech_vector_search import locations


def test_save_and_load_token_data_with_defaults():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [{"word": "a", "id": "0"}, {"word": "b", "id": "1"}]
    with tempfile.TemporaryDirectory() as directory:
        paths = io.save_token_data(embeddings, metadata, directory=directory)
        loaded_embeddings, loaded_metadata = io.load_token_data(
            directory=directory)
    assert paths["embeddings"] == locations.token_embeddings_path(directory)
    assert paths["metadata"] == locations.token_metadata_path(directory)
    assert np.allclose(loaded_embeddings, embeddings)
    assert loaded_metadata == metadata


def test_save_and_load_prototypes_with_name():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [{"word": "a", "subset_id": 0}, {"word": "b", "subset_id": 0}]
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


def test_token_and_prototype_paths_do_not_collide():
    with tempfile.TemporaryDirectory() as directory:
        token_metadata = locations.token_metadata_path(directory)
        prototype_metadata = locations.prototype_metadata_path(directory)
        token_embeddings = locations.token_embeddings_path(directory)
        prototype_vectors = locations.prototype_vectors_path(directory)
    assert token_metadata != prototype_metadata
    assert token_embeddings != prototype_vectors


def test_save_prototypes_raises_when_files_exist():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [{"word": "a", "subset_id": 0}, {"word": "b", "subset_id": 0}]
    with tempfile.TemporaryDirectory() as directory:
        io.save_prototypes(vectors, metadata, directory=directory)
        try:
            io.save_prototypes(vectors, metadata, directory=directory)
        except FileExistsError:
            return
    raise AssertionError("expected FileExistsError when prototype files exist")
