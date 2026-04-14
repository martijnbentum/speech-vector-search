import json

from speech_vector_search import phraser_adapter


UNIT_TYPES = {'phone', 'syllable', 'word', 'phrase'}


class PrototypeMetadata:
    '''metadata for one prototype vector.'''

    def __init__(self, label, unit_type, source_echoframe_keys,
        n_occurrences=None, subset_id=None):
        self.label = label
        self.unit_type = check_unit_type(unit_type)
        self.source_echoframe_keys = list(source_echoframe_keys)
        if n_occurrences is None:
            n_occurrences = len(self.source_echoframe_keys)
        self.n_occurrences = n_occurrences
        self.subset_id = subset_id
        self.validate()

    def __repr__(self):
        return (f'PrototypeMD(label={self.label}, unit={self.unit_type}, '
            f'n={self.n_occurrences})')

    def __str__(self):
        m = self.__repr__()
        m += 'source_echoframe_keys:\n '
        m += f"{'\n'.join(['\t' + x for x in self.source_echoframe_keys])}"
        return m

    def validate(self):
        '''validate one metadata item.'''
        if not isinstance(self.label, str) or not self.label:
            raise ValueError('label must be a non-empty string')
        self.unit_type = check_unit_type(self.unit_type)
        _validate_key_list(self.source_echoframe_keys,
            'source_echoframe_keys')
        if self.n_occurrences != len(self.source_echoframe_keys):
            m = 'n_occurrences must match number of source_echoframe_keys'
            raise ValueError(m)
        if self.subset_id is not None and not isinstance(self.subset_id, int):
            raise ValueError('subset_id must be an int or None')

    def to_dict(self):
        '''return a json-serializable dict.'''
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
        '''save one metadata item as json.'''
        with open(path, 'w') as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_dict(cls, data):
        '''build one metadata item from a dict.'''
        return cls(data['label'], data['unit_type'],
            data['source_echoframe_keys'], n_occurrences=data.get(
            'n_occurrences'),
            subset_id=data.get('subset_id'))

    @classmethod
    def from_json(cls, path):
        '''load one metadata item from json.'''
        with open(path) as handle:
            data = json.load(handle)
        return cls.from_dict(data)


class PrototypeMetadatas:
    '''collection of prototype metadata items.'''

    def __init__(self, items=None, directory=None, name=None, config=None):
        if items is None:
            items = []
        self.items = list(items)
        self.directory = directory
        self.name = name
        self.config = config
        self.validate()

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def append(self, metadata):
        '''append one metadata item.'''
        if not isinstance(metadata, PrototypeMetadata):
            raise ValueError('metadata must be a PrototypeMetadata')
        metadata.validate()
        self.items.append(metadata)

    def validate(self):
        '''validate collection metadata.'''
        if self.name is not None:
            if not isinstance(self.name, str) or not self.name:
                raise ValueError('name must be a non-empty string')
        for item in self.items:
            if not isinstance(item, PrototypeMetadata):
                raise ValueError('all items must be PrototypeMetadata')
            item.validate()

    def to_dicts(self):
        '''return plain dicts for all items.'''
        return [item.to_dict() for item in self.items]

    def save_jsonl(self, path):
        '''save collection as jsonl.'''
        with open(path, 'w') as handle:
            for item in self.items:
                handle.write(json.dumps(item.to_dict()) + '\n')

    @classmethod
    def load_jsonl(cls, path, directory=None, name=None, config=None):
        '''load collection from jsonl.'''
        items = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(PrototypeMetadata.from_dict(json.loads(line)))
        return cls(items, directory=directory, name=name, config=config)


def check_unit_type(unit_type):
    '''check unit type is valid.'''
    return phraser_adapter.resolve_unit_type(unit_type)


def make_prototype_row(label, unit_type, source_echoframe_keys,
    n_occurrences=None, subset_id=None):
    '''build one metadata item.'''
    return PrototypeMetadata(label, unit_type, source_echoframe_keys,
        n_occurrences=n_occurrences, subset_id=subset_id)


def validate_row(metadata):
    '''validate one metadata item.'''
    if not isinstance(metadata, PrototypeMetadata):
        raise ValueError('metadata must be a PrototypeMetadata')
    metadata.validate()


def validate_rows(metadata_items):
    '''validate many metadata items.'''
    PrototypeMetadatas(metadata_items).validate()


def _validate_key_list(values, field_name):
    '''validate one source-key list.'''
    if not isinstance(values, list):
        raise ValueError(f'{field_name} must be a list')
    if not values:
        raise ValueError(f'{field_name} must not be empty')
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f'{field_name} must contain non-empty strings')
