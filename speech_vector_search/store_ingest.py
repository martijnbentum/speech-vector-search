import numpy as np

from speech_vector_search import phraser_adapter
from speech_vector_search import pooling
from speech_vector_search import prototype_artifact


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
        metadata.append(prototype_artifact.make_prototype_row(
            label=resolve_label(row),
            unit_type=phraser_adapter.resolve_unit_type(row),
            source_phraser_keys=[resolve_source_key(row, "phraser_key",
                "source_phraser_keys")],
            source_echoframe_keys=[resolve_source_key(row, "echoframe_key",
                "source_echoframe_keys")],
        ))
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
