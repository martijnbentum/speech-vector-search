import json

import numpy as np

from speech_vector_search import locations
from speech_vector_search import prototype_artifact
from speech_vector_search import utils

def save_metadata_jsonl(rows, path):
    '''save metadata rows to jsonl.
    rows                     metadata records
    path                     jsonl file path
    '''
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def save_prototypes(vectors, metadata, directory=None, name=None, config=None,
    overwrite=False):
    '''save prototype vectors and metadata.
    vectors                  prototype matrix
    metadata                 prototype metadata records
    directory                optional storage directory
    name                     optional base name without extension
    config                   optional config dict to save
    overwrite                whether to overwrite existing files
    '''
    directory = locations.resolve_directory(directory)
    name = locations.resolve_prototype_name(name)
    artifact_directory = locations.prototype_directory(directory, name)
    utils.ensure_directory(artifact_directory)
    vectors_path = locations.prototype_vectors_path(directory, name)
    metadata_path = locations.prototype_metadata_path(directory, name)
    if not overwrite:
        if vectors_path.exists():
            raise FileExistsError(f"vectors file exists: {vectors_path}")
        if metadata_path.exists():
            raise FileExistsError(f"metadata file exists: {metadata_path}")
    prototype_artifact.validate_rows(metadata)
    np.save(vectors_path, np.asarray(vectors, dtype=float))
    save_metadata_jsonl(metadata, metadata_path)
    if config is not None:
        config_path = locations.config_path(directory, name)
        utils.write_json(config, config_path)
    else:
        config_path = None
    return {
        "vectors": vectors_path,
        "metadata": metadata_path,
        "config": config_path,
    }


def load_prototypes(vectors_path=None, metadata_path=None, directory=None,
    name=None):
    '''load saved prototypes.
    vectors_path             optional npy file path
    metadata_path            optional jsonl file path
    directory                optional storage directory
    name                     optional base name without extension
    '''
    vectors_path, metadata_path = resolve_prototype_paths(vectors_path,
        metadata_path, directory, name)
    vectors = np.asarray(np.load(vectors_path), dtype=float)
    metadata = load_metadata_jsonl(metadata_path)
    prototype_artifact.validate_rows(metadata)
    if len(vectors) != len(metadata):
        raise ValueError("metadata length must match number of vectors")
    return vectors, metadata
def load_metadata_jsonl(path):
    '''load metadata rows from jsonl.
    path                    jsonl file path
    '''
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            utils.infer_label_key(row)
            rows.append(row)
    return rows


def resolve_prototype_paths(vectors_path, metadata_path, directory, name):
    '''resolve prototype data paths.
    vectors_path            optional vector file path
    metadata_path           optional metadata file path
    directory               optional storage directory
    name                    optional base name without extension
    '''
    if vectors_path is None:
        vectors_path = locations.prototype_vectors_path(directory, name)
    if metadata_path is None:
        metadata_path = locations.prototype_metadata_path(directory, name)
    return vectors_path, metadata_path
