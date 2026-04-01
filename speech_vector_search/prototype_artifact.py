from speech_vector_search import phraser_adapter
from speech_vector_search import utils


REQUIRED_KEYS = [
    "label",
    "unit_type",
    "source_phraser_keys",
    "source_echoframe_keys",
    "n_occurrences",
]


def make_prototype_row(label, unit_type, source_phraser_keys,
    source_echoframe_keys, n_occurrences=None, **extra):
    '''build one explicit prototype artifact row.
    label                    prototype label
    unit_type                phone, syllable, word, or phrase
    source_phraser_keys      source phraser occurrence keys
    source_echoframe_keys    source echoframe occurrence keys
    n_occurrences            optional occurrence count
    '''
    row = {
        "label": label,
        "unit_type": phraser_adapter.resolve_unit_type(unit_type),
        "source_phraser_keys": list(source_phraser_keys),
        "source_echoframe_keys": list(source_echoframe_keys),
    }
    if n_occurrences is None:
        n_occurrences = len(row["source_phraser_keys"])
    row["n_occurrences"] = n_occurrences
    row.update(extra)
    validate_row(row)
    return row


def validate_rows(rows):
    '''validate stored prototype metadata rows.
    rows                     prototype metadata records
    '''
    for row in rows:
        validate_row(row)


def validate_row(row):
    '''validate one prototype metadata row.
    row                      prototype metadata record
    '''
    for key in REQUIRED_KEYS:
        if key not in row:
            raise ValueError(f"prototype metadata row must contain '{key}'")
    utils.infer_label_key(row)
    phraser_adapter.resolve_unit_type(row["unit_type"])
    validate_source_list(row["source_phraser_keys"], "source_phraser_keys")
    validate_source_list(row["source_echoframe_keys"],
        "source_echoframe_keys")
    if row["n_occurrences"] != len(row["source_phraser_keys"]):
        raise ValueError(
            "n_occurrences must match number of source_phraser_keys"
        )
    if len(row["source_phraser_keys"]) != len(row["source_echoframe_keys"]):
        raise ValueError(
            "source_phraser_keys and source_echoframe_keys must match in length"
        )


def validate_source_list(values, field_name):
    '''validate one list of source identifiers.
    values                   field value to validate
    field_name               metadata field name
    '''
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list")
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    for value in values:
        if not value:
            raise ValueError(f"{field_name} must not contain empty values")
