import json

from speech_vector_search import metadata


def test_vector_metadata_dict_roundtrip():
    item = metadata.VectorMetadata(
        'hello',
        'word',
        [b'\xe0', b'\xe1'],
        subset_id=4,
    )
    loaded = metadata.VectorMetadata.from_dict(item.to_dict())
    assert loaded.label == 'hello'
    assert loaded.unit_type == 'word'
    assert loaded.source_echoframe_keys == [b'\xe0', b'\xe1']
    assert loaded.subset_id == 4


def test_vector_metadatas_reject_mixed_unit_types():
    items = [
        metadata.VectorMetadata('hello', 'word', [b'\xe0']),
        metadata.VectorMetadata('hh', 'phone', [b'\xe1']),
    ]
    try:
        metadata.VectorMetadatas(items, name='demo')
    except ValueError as exc:
        assert 'same unit type' in str(exc)
        return
    raise AssertionError('expected ValueError for mixed unit types')


def test_vector_metadatas_save_and_load_json(tmp_path):
    directory = str(tmp_path)
    items = [
        metadata.VectorMetadata('hello', 'word', [b'\xe0']),
        metadata.VectorMetadata('world', 'word', [b'\xe1']),
    ]
    rows = metadata.VectorMetadatas(items, directory=directory, name='demo')
    rows.save_json()
    loaded = metadata.VectorMetadatas.load_json(tmp_path / 'demo.json')
    assert len(loaded) == 2
    assert loaded.name == 'demo'
    assert loaded.directory == directory
    assert loaded[0].to_dict() == items[0].to_dict()
    assert loaded[1].to_dict() == items[1].to_dict()


def test_vector_metadatas_save_json_skips_existing_file(tmp_path, capsys):
    items = [metadata.VectorMetadata('hello', 'word', [b'\xe0'])]
    rows = metadata.VectorMetadatas(
        items,
        directory=str(tmp_path),
        name='demo',
    )
    rows.save_json()
    rows.save_json()
    captured = capsys.readouterr()
    assert 'already exists' in captured.out


def test_vector_metadatas_load_json_rejects_missing_file(tmp_path):
    missing = tmp_path / 'missing.json'
    try:
        metadata.VectorMetadatas.load_json(missing)
    except FileNotFoundError as exc:
        assert 'does not exist' in str(exc)
        return
    raise AssertionError('expected FileNotFoundError for missing file')


def test_vector_metadatas_load_json_warns_on_path_mismatch(tmp_path, capsys):
    directory = str(tmp_path / 'stored')
    items = [metadata.VectorMetadata('hello', 'word', [b'\xe0'])]
    rows = metadata.VectorMetadatas(items, directory=directory, name='demo')
    other_path = tmp_path / 'other.json'
    other_path.write_text(json.dumps({
        'directory': directory,
        'name': 'demo',
        'metadatas': [rows[0].to_dict()],
    }))
    loaded = metadata.VectorMetadatas.load_json(other_path)
    captured = capsys.readouterr()
    assert 'does not match stored' in captured.out
    assert loaded.directory == directory
    assert loaded.name == 'demo'


def test_vector_metadatas_repr_includes_summary_fields():
    items = [
        metadata.VectorMetadata('hello', 'word', [b'\xe0']),
        metadata.VectorMetadata('world', 'word', [b'\xe1']),
    ]
    rows = metadata.VectorMetadatas(items, directory='artifacts', name='demo')
    text = repr(rows)
    assert 'VectorMetadatas' in text
    assert 'demo' in text
    assert 'unit=word' in text
    assert 'n labels=2' in text


def test_vector_metadata_rejects_empty_label():
    try:
        metadata.VectorMetadata('', 'word', [b'\xe0'])
    except ValueError as exc:
        assert 'non-empty string' in str(exc)
        return
    raise AssertionError('expected ValueError for empty label')


def test_vector_metadata_rejects_empty_unit_type():
    try:
        metadata.VectorMetadata('hello', '', [b'\xe0'])
    except ValueError as exc:
        assert 'non-empty string' in str(exc)
        return
    raise AssertionError('expected ValueError for empty unit_type')


def test_vector_metadata_rejects_empty_source_keys():
    try:
        metadata.VectorMetadata('hello', 'word', [])
    except ValueError as exc:
        assert 'must not be empty' in str(exc)
        return
    raise AssertionError('expected ValueError for empty source_echoframe_keys')


def test_vector_metadata_rejects_non_bytes_source_keys():
    try:
        metadata.VectorMetadata('hello', 'word', ['e0'])
    except ValueError as exc:
        assert 'must contain bytes' in str(exc)
        return
    raise AssertionError('expected ValueError for non-bytes source_echoframe_keys')


def test_vector_metadata_rejects_non_integer_subset_id():
    try:
        metadata.VectorMetadata('hello', 'word', [b'\xe0'], subset_id='0')
    except ValueError as exc:
        assert 'int or None' in str(exc)
        return
    raise AssertionError('expected ValueError for invalid subset_id')


def test_vector_metadata_repr_is_concise():
    item = metadata.VectorMetadata('hello', 'word', [b'\xe0', b'\xe1'])
    text = repr(item)
    assert 'VectorMD' in text
    assert 'hello' in text
    assert 'unit=word' in text
    assert 'source_echoframe_keys' not in text
