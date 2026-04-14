import tempfile

from speech_vector_search import metadata
from speech_vector_search import prototypes


def test_make_prototype_row_sets_occurrence_count():
    item = prototypes.make_prototype_row('a', 3, [
        {
            'unit_type': 'word',
            'echoframe_key': 'e0',
        },
        {
            'unit_type': 'word',
            'echoframe_key': 'e1',
        },
    ])
    assert item.n_occurrences == 2
    assert item.subset_id == 3
    assert item.source_echoframe_keys == ['e0', 'e1']
    assert not hasattr(item, 'source_phraser_keys')


def test_prototype_metadata_to_dict_includes_subset_id():
    item = metadata.PrototypeMetadata(
        'hello',
        'word',
        ['e0', 'e1'],
        subset_id=4,
    )
    assert item.to_dict() == {
        'label': 'hello',
        'unit_type': 'word',
        'source_echoframe_keys': ['e0', 'e1'],
        'n_occurrences': 2,
        'subset_id': 4,
    }


def test_prototype_metadata_json_roundtrip():
    item = metadata.PrototypeMetadata(
        'hello',
        'word',
        ['e0'],
        subset_id=0,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = metadata_path = f'{directory}/item.json'
        item.to_json(metadata_path)
        loaded = metadata.PrototypeMetadata.from_json(path)
    assert loaded.label == 'hello'
    assert loaded.subset_id == 0
    assert loaded.source_echoframe_keys == ['e0']


def test_prototype_metadata_repr_and_str_are_small_and_verbose():
    item = metadata.PrototypeMetadata('hello', 'word', ['e0', 'e1'])
    assert 'source_echoframe_keys' not in repr(item)
    assert 'source_echoframe_keys' in str(item)
    assert 'e0' in str(item)


def test_validate_rows_rejects_mismatched_source_lengths():
    items = [
        metadata.PrototypeMetadata(
            'a',
            'word',
            ['e0'],
        )
    ]
    items[0].source_echoframe_keys.append('e1')
    try:
        prototypes.validate_rows(items)
    except ValueError as exc:
        assert 'n_occurrences' in str(exc)
        return
    raise AssertionError('expected ValueError for mismatched source lengths')
