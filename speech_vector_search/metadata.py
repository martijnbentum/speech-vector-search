import json


SUPPORTED_UNIT_TYPES = ("phone", "syllable", "word", "phrase")

UNIT_TYPE_ALIASES = {
    "phoneme": "phone",
    "phones": "phone",
    "syllables": "syllable",
    "words": "word",
    "phrases": "phrase",
}


def resolve_unit_type(value):
    '''resolve one supported unit type string.'''
    if isinstance(value, dict):
        value = value.get("unit_type", "word")
    elif hasattr(value, 'unit_type'):
        value = value.unit_type
    if value is None:
        value = "word"
    if not isinstance(value, str):
        raise ValueError("unit_type must be a string")
    value = value.strip().lower()
    if value in UNIT_TYPE_ALIASES:
        value = UNIT_TYPE_ALIASES[value]
    if value not in SUPPORTED_UNIT_TYPES:
        raise ValueError(f"unsupported unit_type: {value!r}")
    return value


def resolve_shared_unit_type(rows):
    '''resolve one unit type shared by all rows, raising if they differ.'''
    unit_types = [resolve_unit_type(row) for row in rows]
    unique = sorted(set(unit_types))
    if len(unique) != 1:
        raise ValueError("rows must share one unit_type")
    return unique[0]


class PrototypeMetadata:
    '''metadata for one prototype vector.'''

    def __init__(self, label, unit_type, source_echoframe_keys,
        n_occurrences=None, subset_id=None):
        self.label = label
        self.unit_type = resolve_unit_type(unit_type)
        self.source_echoframe_keys = list(source_echoframe_keys)
        if n_occurrences is None:
            n_occurrences = len(self.source_echoframe_keys)
        self.n_occurrences = n_occurrences
        self.subset_id = subset_id
        self._validate()

    def __repr__(self):
        return (f'PrototypeMD(label={self.label}, unit={self.unit_type}, '
            f'n={self.n_occurrences})')

    def _validate(self):
        if not isinstance(self.label, str) or not self.label:
            raise ValueError('label must be a non-empty string')
        self.unit_type = resolve_unit_type(self.unit_type)
        _validate_key_list(self.source_echoframe_keys, 'source_echoframe_keys')
        if self.n_occurrences != len(self.source_echoframe_keys):
            raise ValueError(
                'n_occurrences must match number of source_echoframe_keys')
        if self.subset_id is not None and not isinstance(self.subset_id, int):
            raise ValueError('subset_id must be an int or None')

    def to_dict(self):
        data = {
            'label': self.label,
            'unit_type': self.unit_type,
            'source_echoframe_keys': list(self.source_echoframe_keys),
            'n_occurrences': self.n_occurrences,
        }
        if self.subset_id is not None:
            data['subset_id'] = self.subset_id
        return data

    def to_json(self, path):
        with open(path, 'w') as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_dict(cls, data):
        return cls(data['label'], data['unit_type'],
            data['source_echoframe_keys'],
            n_occurrences=data.get('n_occurrences'),
            subset_id=data.get('subset_id'))

    @classmethod
    def from_json(cls, path):
        with open(path) as handle:
            data = json.load(handle)
        return cls.from_dict(data)


class PrototypeMetadatas:
    '''collection of PrototypeMetadata items.'''

    def __init__(self, items=None):
        if items is None:
            items = []
        self.items = list(items)
        self._validate()

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def append(self, item):
        if not isinstance(item, PrototypeMetadata):
            raise ValueError('item must be a PrototypeMetadata')
        item._validate()
        self.items.append(item)

    def _validate(self):
        for item in self.items:
            if not isinstance(item, PrototypeMetadata):
                raise ValueError('all items must be PrototypeMetadata')
            item._validate()

    def save_jsonl(self, path):
        with open(path, 'w') as handle:
            for item in self.items:
                handle.write(json.dumps(item.to_dict()) + '\n')

    @classmethod
    def load_jsonl(cls, path):
        items = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(PrototypeMetadata.from_dict(json.loads(line)))
        return cls(items)


def validate_rows(metadata_items):
    '''validate a list of PrototypeMetadata items.'''
    PrototypeMetadatas(metadata_items)._validate()


def _validate_key_list(values, field_name):
    if not isinstance(values, list):
        raise ValueError(f'{field_name} must be a list')
    if not values:
        raise ValueError(f'{field_name} must not be empty')
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f'{field_name} must contain non-empty strings')
