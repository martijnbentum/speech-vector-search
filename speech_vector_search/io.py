import numpy as np

from speech_vector_search import locations
from speech_vector_search import prototype_artifact
from speech_vector_search import utils


def save_prototypes(vectors, metadata, name=None, directory=None, config=None,
    overwrite=False):
    '''save prototype vectors and metadata.
    vectors                  prototype matrix
    metadata                 prototype metadata records
    name                     base name without extension
    directory                optional storage directory
    config                   optional config dict to save
    overwrite                whether to overwrite existing files
    '''
    vectors_path = locations.make_path(directory, name, 'vectors', overwrite)
    metadata_path = locations.make_path(directory, name, 'metadata', overwrite)
    prototype_artifact.validate_rows(metadata)
    np.save(vectors_path, np.asarray(vectors, dtype=float))
    utils.save_jsonl(metadata, metadata_path)
    config_path = _handle_config(config, name, directory, overwrite)
    return {'vectors': vectors_path, 'metadata': metadata_path,
        'config': config_path}


def load_prototypes(name=None, directory=None):
    '''load saved prototypes.
    name                     base name without extension
    directory                optional storage directory
    '''
    vectors_path = locations.make_path(directory, name, 'vectors',load = True)
    metadata_path = locations.make_path(directory, name, 'metadata',load = True)
    vectors = np.asarray(np.load(vectors_path), dtype=float)
    metadata = utils.load_jsonl(metadata_path)
    prototype_artifact.validate_rows(metadata)
    if len(vectors) != len(metadata):
        m  = 'metadata length must match number of vectors'
        m += f' ({len(metadata)} vs {len(vectors)})'
        m += f' in {metadata_path} and {vectors_path}'
        raise ValueError(m)
    return vectors, metadata


def _handle_config(config, name, directory, overwrite):
    if config is None:
        return None
    config_path = locations.make_path(directory, name, 'config', overwrite)
    utils.write_json(config, config_path)
    return config_path
