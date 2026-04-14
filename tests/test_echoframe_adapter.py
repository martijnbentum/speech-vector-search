import numpy as np

from speech_vector_search import echoframe_adapter
from speech_vector_search import prototypes


class TokenEmbeddings:
    def __init__(self, vectors, echoframe_keys, dims=(), layers=None):
        self._vectors = vectors
        self.echoframe_keys = list(echoframe_keys)
        self.dims = tuple(dims)
        self.layers = layers

    def to_numpy(self):
        return self._vectors


def test_build_mean_prototype_from_token_embeddings_matches_numpy_path(
    monkeypatch,
):
    monkeypatch.setattr(echoframe_adapter, "TokenEmbeddings", TokenEmbeddings)
    token_embeddings = TokenEmbeddings(
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        ["e0", "e1", "e2"],
    )

    vector, row = echoframe_adapter.build_mean_prototype_from_token_embeddings(
        token_embeddings,
        label="bake",
        unit_type="word",
        subset_id=3,
    )

    expected_vector, expected_row = prototypes.build_mean_prototype(
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        [
            {"label": "bake", "unit_type": "word", "echoframe_key": "e0"},
            {"label": "bake", "unit_type": "word", "echoframe_key": "e1"},
            {"label": "bake", "unit_type": "word", "echoframe_key": "e2"},
        ],
        subset_id=3,
        label="bake",
    )

    assert np.allclose(vector, expected_vector)
    assert row.to_dict() == expected_row.to_dict()


def test_build_mean_prototype_from_token_embeddings_collapses_one_layer(
    monkeypatch,
):
    monkeypatch.setattr(echoframe_adapter, "TokenEmbeddings", TokenEmbeddings)
    token_embeddings = TokenEmbeddings(
        np.array([
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]),
        ["e0", "e1"],
        dims=("tokens", "layers"),
        layers=[0],
    )

    vector, row = echoframe_adapter.build_mean_prototype_from_token_embeddings(
        token_embeddings,
        label="bake",
        unit_type="word",
    )

    assert np.allclose(vector, [1.0, 0.0])
    assert row.source_echoframe_keys == ["e0", "e1"]


def test_build_subset_prototypes_from_token_embeddings_matches_numpy_path(
    monkeypatch,
):
    monkeypatch.setattr(echoframe_adapter, "TokenEmbeddings", TokenEmbeddings)
    token_embeddings = TokenEmbeddings(
        np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]),
        ["e0", "e1", "e2", "e3"],
    )

    vectors, rows, config = (
        echoframe_adapter.build_subset_prototypes_from_token_embeddings(
            token_embeddings,
            label="bake",
            unit_type="word",
            subset_size=2,
            n_subsets=2,
            seed=0,
        )
    )

    expected_vectors, expected_rows, expected_config = (
        prototypes.build_subset_prototypes(
            "bake",
            np.array([
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]),
            [
                {"label": "bake", "unit_type": "word", "echoframe_key": "e0"},
                {"label": "bake", "unit_type": "word", "echoframe_key": "e1"},
                {"label": "bake", "unit_type": "word", "echoframe_key": "e2"},
                {"label": "bake", "unit_type": "word", "echoframe_key": "e3"},
            ],
            subset_size=2,
            n_subsets=2,
            seed=0,
        )
    )

    assert np.allclose(vectors, expected_vectors)
    assert [row.to_dict() for row in rows] == [row.to_dict()
        for row in expected_rows]
    assert config == expected_config


def test_build_mean_prototype_from_token_embeddings_rejects_frames(
    monkeypatch,
):
    monkeypatch.setattr(echoframe_adapter, "TokenEmbeddings", TokenEmbeddings)
    token_embeddings = TokenEmbeddings(
        np.array([[1.0, 0.0]]),
        ["e0"],
        dims=("tokens", "frames"),
    )
    try:
        echoframe_adapter.build_mean_prototype_from_token_embeddings(
            token_embeddings,
            label="bake",
            unit_type="word",
        )
    except ValueError as exc:
        assert "frames" in str(exc)
        return
    raise AssertionError("expected ValueError for frame-major token embeddings")


def test_build_mean_prototype_from_token_embeddings_rejects_multiple_layers(
    monkeypatch,
):
    monkeypatch.setattr(echoframe_adapter, "TokenEmbeddings", TokenEmbeddings)
    token_embeddings = TokenEmbeddings(
        np.array([
            [[1.0, 0.0], [2.0, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
        ]),
        ["e0", "e1"],
        dims=("tokens", "layers"),
        layers=[0, 1],
    )
    try:
        echoframe_adapter.build_mean_prototype_from_token_embeddings(
            token_embeddings,
            label="bake",
            unit_type="word",
        )
    except ValueError as exc:
        assert "one layer" in str(exc)
        return
    raise AssertionError("expected ValueError for multi-layer token embeddings")


def test_build_mean_prototype_from_token_embeddings_rejects_ragged_token_arrays(
    monkeypatch,
):
    monkeypatch.setattr(echoframe_adapter, "TokenEmbeddings", TokenEmbeddings)
    token_embeddings = TokenEmbeddings(
        np.array(
            [
                np.array([1.0, 0.0], dtype=float),
                np.array([1.0], dtype=float),
            ],
            dtype=object,
        ),
        ["e0", "e1"],
    )
    try:
        echoframe_adapter.build_mean_prototype_from_token_embeddings(
            token_embeddings,
            label="bake",
            unit_type="word",
        )
    except ValueError as exc:
        assert "one shape" in str(exc)
        return
    raise AssertionError("expected ValueError for ragged token arrays")
