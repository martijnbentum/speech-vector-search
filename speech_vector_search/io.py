import json
from pathlib import Path

import numpy as np

from speech_vector_search import metadata as metadata_module
from speech_vector_search import prototypes


DEFAULT_STORAGE_DIR = Path("data")
DEFAULT_PROTOTYPE_NAME = "prototypes"


def save_prototypes(vectors, metadata, name=None, directory=None, config=None,
    overwrite=False):
    '''save prototype vectors and metadata to disk.
    vectors                  prototype matrix
    metadata                 list of PrototypeMetadata
    name                     base filename (no extension)
    directory                storage directory (default: data/)
    config                   optional config dict
    overwrite                overwrite existing files
    '''
    vectors_path = _make_path(directory, name, 'vectors', overwrite)
    metadata_path = _make_path(directory, name, 'metadata', overwrite)
    prototypes.validate_rows(metadata)
    np.save(vectors_path, np.asarray(vectors, dtype=float))
    metadata_module.PrototypeMetadatas(metadata).save_jsonl(metadata_path)
    config_path = _save_config(config, name, directory, overwrite)
    return {'vectors': vectors_path, 'metadata': metadata_path,
        'config': config_path}


def load_prototypes(name=None, directory=None):
    '''load prototype vectors and metadata from disk.
    name                     base filename (no extension)
    directory                storage directory (default: data/)
    '''
    vectors_path = _make_path(directory, name, 'vectors', load=True)
    metadata_path = _make_path(directory, name, 'metadata', load=True)
    vectors = np.asarray(np.load(vectors_path), dtype=float)
    metadatas = metadata_module.PrototypeMetadatas.load_jsonl(metadata_path)
    metadata_items = metadatas.items
    prototypes.validate_rows(metadata_items)
    if len(vectors) != len(metadata_items):
        raise ValueError(
            f'metadata length must match number of vectors '
            f'({len(metadata_items)} vs {len(vectors)}) '
            f'in {metadata_path} and {vectors_path}')
    return vectors, metadata_items


def _make_path(directory, name, suffix, overwrite=False, load=False):
    directory = Path(directory) if directory is not None else DEFAULT_STORAGE_DIR
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = DEFAULT_PROTOTYPE_NAME
    path = directory / f'{name}_{suffix}'
    if suffix == 'vectors':
        path = path.with_suffix('.npy')
    elif suffix == 'metadata':
        path = path.with_suffix('.jsonl')
    elif suffix == 'config':
        path = path.with_suffix('.json')
    else:
        raise ValueError(f"suffix must be 'vectors', 'metadata', or 'config'")
    if load:
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
    elif not overwrite and path.exists():
        raise FileExistsError(f"file exists: {path}")
    return path


def _save_config(config, name, directory, overwrite):
    if config is None:
        return None
    path = _make_path(directory, name, 'config', overwrite)
    with open(path, 'w') as handle:
        json.dump(config, handle, indent=2)
    return path
