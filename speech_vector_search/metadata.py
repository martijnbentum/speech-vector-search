import json
from pathlib import Path

class VectorMetadatas:
    '''collection of VectorMetadata items.'''
    def __init__(self, metadatas, name, directory= 'data'):
        self.metadatas = metadatas
        self.directory = directory
        self.name = name
        self._validate()
        self._set_info()

    def __len__(self):
        return len(self.metadatas)

    def __repr__(self):
        m = f'VectorMetadatas(name={self.name}, n={len(self.metadatas)}, '
        m += f'unit={self.unit_type}, n labels={len(self.labels)})'
        return m

    def _set_info(self):
        self.path = make_metadata_path(self.directory, self.name)
        self.labels = set(metadata.label for metadata in self.metadatas)
        self.unit_type = self.metadatas[0].unit_type

    def __getitem__(self, index):
        return self.metadatas[index]

    def to_dict(self):
        d = {'directory': self.directory, 'name': self.name,
            'metadatas': [metadata.to_dict() for metadata in self.metadatas],}
        return d

    def save_json(self, overwrite=False):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            print(f'{self.path} already exists')
            return
        with open(self.path, 'w') as fout:
            json.dump(self.to_dict(), fout)

    @property
    def stored(self):
        '''check if the metadata file exists at the expected path.'''
        return self.path.exists()

    @classmethod
    def load_json(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'{path} does not exist')
        metadatas = []
        with open(path, 'r') as fin:
            data = json.load(fin)
        for md in data['metadatas']:
            metadatas.append(VectorMetadata.from_dict(md))
        directory = data['directory']
        name = data['name']
        p = Path(directory) / f'{name}.json'
        if p != path:
            print(f'Warning: path {path} does not match stored: {p}')
            print(f'loaded object will use this path if you save: {p}')
        return cls(metadatas, directory=directory, name=name)

    def _validate(self):
        if not isinstance(self.metadatas, list):
            raise ValueError('metadatas must be a list')
        if not self.metadatas:
            raise ValueError('metadatas must not be empty')
        if any(not isinstance(md, VectorMetadata) for md in self.metadatas):
            raise ValueError('all items must be VectorMetadata')
        unit_types = {md.unit_type for md in self.metadatas}
        if len(unit_types) != 1:
            raise ValueError('all metadatas must share the same unit type')
        


class VectorMetadata:
    '''metadata for one speech vector.'''
    def __init__(self, label, unit_type, source_echoframe_keys, subset_id=None):
        self.label = label
        self.unit_type = unit_type
        self.source_echoframe_keys = list(source_echoframe_keys)
        self.n_occurrences = len(self.source_echoframe_keys)
        self.subset_id = subset_id
        self._validate()

    def __repr__(self):
        m = f'VectorMD(label={self.label}, '
        m += f'unit={self.unit_type}, n={self.n_occurrences})'
        return m
            

    def to_dict(self):
        data = {'label': self.label, 'unit_type': self.unit_type,
            'source_echoframe_keys_hex': self.source_echoframe_keys_hex,
            'subset_id': self.subset_id,}
        return data

    @property
    def source_echoframe_keys_hex(self):
        '''source echoframe keys as hex strings.'''
        return [key.hex() for key in self.source_echoframe_keys]

    @classmethod
    def from_dict(cls, data):
        label = data['label']
        unit_type = data['unit_type']
        hex_keys = data['source_echoframe_keys_hex']
        byte_keys = [bytes.fromhex(hex_key) for hex_key in hex_keys]
        echoframe_keys = byte_keys
        subset_id = data['subset_id']
        return cls(label, unit_type, echoframe_keys, subset_id=subset_id)

    def _validate(self):
        if not isinstance(self.label, str) or not self.label:
            raise ValueError('label must be a non-empty string')
        if not isinstance(self.unit_type, str) or not self.unit_type:
            raise ValueError('unit_type must be a non-empty string')
        if not isinstance(self.source_echoframe_keys, list):
            raise ValueError('source_echoframe_keys must be a list')
        if not self.source_echoframe_keys:
            raise ValueError('source_echoframe_keys must not be empty')
        if any(not isinstance(key, (bytes, bytearray))
            for key in self.source_echoframe_keys):
            raise ValueError('source_echoframe_keys must contain bytes')
        if self.subset_id is not None and not isinstance(self.subset_id, int):
            raise ValueError('subset_id must be an int or None')


def make_metadata_path(directory, name, extension='.json'):
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    path = Path(directory) / f'{name}{extension}'
    return path


