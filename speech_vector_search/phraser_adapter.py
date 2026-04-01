SUPPORTED_UNIT_TYPES = ("phone", "syllable", "word", "phrase")

UNIT_TYPE_ALIASES = {
    "phoneme": "phone",
    "phones": "phone",
    "syllables": "syllable",
    "words": "word",
    "phrases": "phrase",
}


def resolve_unit_type(value):
    '''resolve one supported unit type.
    value                    unit-type string or metadata row
    '''
    if isinstance(value, dict):
        value = value.get("unit_type", "word")
    if value is None:
        value = "word"
    if not isinstance(value, str):
        raise ValueError("unit_type must be a string")
    value = value.strip().lower()
    if value in UNIT_TYPE_ALIASES:
        value = UNIT_TYPE_ALIASES[value]
    if value not in SUPPORTED_UNIT_TYPES:
        raise ValueError(
            f"unsupported unit_type: {value}"
        )
    return value


def resolve_shared_unit_type(rows):
    '''resolve one shared unit type from many rows.
    rows                     metadata rows
    '''
    unit_types = [resolve_unit_type(row) for row in rows]
    unique = sorted(set(unit_types))
    if len(unique) != 1:
        raise ValueError("rows must share one unit_type")
    return unique[0]
