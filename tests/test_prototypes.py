import numpy as np

from speech_vector_search import normalize
from speech_vector_search import prototypes


def test_build_mean_prototype():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    metadata = [
        {'label': 'bake', 'echoframe_key': 'e0', 'unit_type': 'word'},
        {'label': 'bake', 'echoframe_key': 'e1', 'unit_type': 'word'},
        {'label': 'bake', 'echoframe_key': 'e2', 'unit_type': 'word'},
    ]
    vector, row = prototypes.build_mean_prototype(embeddings, metadata)
    assert vector.shape == (2,)
    assert np.allclose(vector, [1.0, 0.0])
    assert row.label == 'bake'
    assert row.unit_type == 'word'
    assert row.source_echoframe_keys == ['e0', 'e1', 'e2']
    assert row.n_occurrences == 3
    assert not hasattr(row, 'source_phraser_keys')


def test_build_subset_prototypes():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    metadata = [
        {"label": "bake", "echoframe_key": "e0", "unit_type": "word"},
        {"label": "bake", "echoframe_key": "e1", "unit_type": "word"},
        {"label": "bake", "echoframe_key": "e2", "unit_type": "word"},
        {"label": "bake", "echoframe_key": "e3", "unit_type": "word"},
    ]
    vectors, rows, config = prototypes.build_subset_prototypes('bake',
        embeddings, metadata, subset_size=2, n_subsets=2, seed=0)
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], [1.0, 0.0])
    assert rows[0].label == 'bake'
    assert rows[0].unit_type == 'word'
    assert sorted(rows[0].source_echoframe_keys) in (
        ['e0', 'e1'],
        ['e2', 'e3'],
    )
    assert rows[0].n_occurrences == 2
    assert not hasattr(rows[0], 'source_phraser_keys')
    assert config["prototype_method"] == "subset_mean"
    assert config["strict_non_overlapping"] is True


def test_l2_normalize():
    vector = np.array([3.0, 4.0])
    normalized = normalize.l2_normalize(vector)
    assert np.allclose(normalized, [0.6, 0.8])


def test_build_subset_prototypes_rejects_mixed_labels():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    metadata = [
        {"label": "bake", "echoframe_key": "e0", "unit_type": "word"},
        {"label": "cake", "echoframe_key": "e1", "unit_type": "word"},
    ]
    try:
        prototypes.build_subset_prototypes('bake', embeddings, metadata,
            subset_size=1, n_subsets=2, seed=0)
    except ValueError as exc:
        assert "exactly one label" in str(exc)
        return
    raise AssertionError("expected ValueError for mixed labels")


def test_make_config():
    config = prototypes.make_config(3, 2, 6, 7, True, label="bake")
    assert config == {
        "prototype_method": "subset_mean",
        "subset_size": 3,
        "n_subsets": 2,
        "min_count": 6,
        "seed": 7,
        "strict_non_overlapping": True,
        "label": "bake",
    }


def test_build_subset_prototypes_non_strict_config_min_count():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    metadata = [
        {'label': 'bake', 'echoframe_key': 'e0', 'unit_type': 'word'},
        {'label': 'bake', 'echoframe_key': 'e1', 'unit_type': 'word'},
        {'label': 'bake', 'echoframe_key': 'e2', 'unit_type': 'word'},
    ]
    _, rows, config = prototypes.build_subset_prototypes('bake', embeddings,
        metadata, subset_size=3, n_subsets=2, seed=0,
        strict_non_overlapping=False)
    assert len(rows) == 1
    assert config['min_count'] == 3
