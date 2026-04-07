import numpy as np

from speech_vector_search import phraser_adapter
from speech_vector_search import pooling
from speech_vector_search import prototypes


def load_source_occurrences(source, unit_type=None):
    '''load echoframe-style occurrence rows from a source.
    source                   iterable rows or store object
    unit_type                optional unit-type filter
    '''
    rows = _coerce_source_rows(source, unit_type=unit_type)
    if unit_type is None:
        return rows
    resolved = phraser_adapter.resolve_unit_type(unit_type)
    return [
        row for row in rows
        if phraser_adapter.resolve_unit_type(row) == resolved
    ]


def extract_echoframe_metadata(store, label, unit_type=None):
    '''extract occurrence metadata rows for one label-tagged echoframe set.
    store                    echoframe-style store with find_by_tag
    label                    label stored as an echoframe tag
    unit_type                optional unit type to attach to each row
    '''
    rows = []
    for metadata in store.find_by_tag(label):
        rows.append({
            "label": label,
            "unit_type": resolve_metadata_unit_type(metadata, unit_type),
            "phraser_key": metadata.phraser_key,
            "echoframe_key": metadata.entry_id,
            "echoframe_metadata": metadata,
        })
    return rows


def load_echoframe_payloads(store, metadata_rows, frame_key="frames"):
    '''load payloads for echoframe metadata rows.
    store                    echoframe-style store with a storage loader
    metadata_rows            rows containing echoframe metadata references
    frame_key                output key for the loaded payload
    '''
    rows = []
    for row in metadata_rows:
        loaded = dict(row)
        metadata = _resolve_echoframe_metadata(row)
        loaded[frame_key] = store.storage.load(metadata)
        rows.append(loaded)
    return rows


def build_prototype_artifacts(source, pooling_method="mean", unit_type=None,
    frame_key="frames"):
    '''convert source occurrences into prototype artifacts.
    source                   iterable rows or store object
    pooling_method           frame reduction method
    unit_type                optional unit-type filter
    frame_key                row key containing frame vectors
    '''
    rows = load_source_occurrences(source, unit_type=unit_type)
    vectors = []
    metadata = []
    for row in rows:
        vectors.append(pooling.pool_frames(row[frame_key],
            method=pooling_method))
        metadata.append(prototypes.make_prototype_row(resolve_label(row), None,
            [{
                'unit_type': phraser_adapter.resolve_unit_type(row),
                'phraser_key': resolve_source_key(row, 'phraser_key',
                    'source_phraser_keys'),
                'echoframe_key': resolve_source_key(row, 'echoframe_key',
                    'source_echoframe_keys'),
            }]))
    if vectors:
        vectors = np.vstack(vectors)
    else:
        vectors = np.zeros((0, 0), dtype=float)
    config = {
        "prototype_method": "pooled_occurrence",
        "pooling_method": pooling_method,
        "unit_type": (
            None if unit_type is None
            else phraser_adapter.resolve_unit_type(unit_type)
        ),
    }
    return vectors, metadata, config


def _coerce_source_rows(source, unit_type=None):
    '''normalize supported source inputs to a list of rows.
    source                   iterable rows or store object
    unit_type                optional unit-type filter
    '''
    if hasattr(source, "load_occurrences"):
        if unit_type is None:
            rows = source.load_occurrences()
        else:
            rows = source.load_occurrences(unit_type=unit_type)
    elif hasattr(source, "occurrences"):
        rows = source.occurrences
    else:
        rows = source
    return list(rows)


def _resolve_echoframe_metadata(row):
    '''resolve one echoframe metadata object from a row.'''
    if hasattr(row, "entry_id") and hasattr(row, "phraser_key"):
        return row
    if "echoframe_metadata" in row:
        return row["echoframe_metadata"]
    raise ValueError("row must contain 'echoframe_metadata'")


def resolve_metadata_unit_type(metadata, unit_type=None):
    '''resolve a unit type for echoframe metadata rows.
    metadata                 echoframe metadata object or row
    unit_type                optional explicit unit type override
    '''
    if unit_type is not None:
        return phraser_adapter.resolve_unit_type(unit_type)
    if isinstance(metadata, dict) and 'unit_type' in metadata:
        return phraser_adapter.resolve_unit_type(metadata['unit_type'])
    if hasattr(metadata, 'unit_type'):
        return phraser_adapter.resolve_unit_type(metadata.unit_type)
    raise ValueError('unit_type is required when metadata lacks one')


def resolve_label(row):
    '''resolve the label for one source occurrence.
    row                      occurrence metadata row
    '''
    for key in ("label", "word", "text"):
        if key in row and row[key]:
            return row[key]
    raise ValueError("source occurrence must contain a label")


def resolve_source_key(row, singular_key, plural_key):
    '''resolve one source key from one occurrence row.
    row                      occurrence metadata row
    singular_key             single-source field name
    plural_key               list field name
    '''
    if singular_key in row:
        value = row[singular_key]
    elif plural_key in row:
        values = row[plural_key]
        if len(values) != 1:
            raise ValueError(f"{plural_key} must contain exactly one value")
        value = values[0]
    elif singular_key == "echoframe_key" and "id" in row:
        value = row["id"]
    else:
        raise ValueError(f"source occurrence must contain '{singular_key}'")
    if not value:
        raise ValueError(f"{singular_key} must not be empty")
    return value
