import csv
import json

import numpy as np

from speech_vector_search import locations
from speech_vector_search import utils


def load_token_data(embeddings_path=None, metadata_path=None, directory=None,
    name=None):
    '''load token embeddings and metadata.
    embeddings_path         optional npz or npy file path
    metadata_path           optional jsonl or csv file path
    directory               optional storage directory
    name                    optional base name without extension
    '''
    embeddings_path, metadata_path = resolve_token_paths(embeddings_path,
        metadata_path, directory, name)
    embeddings_path = locations.resolve_directory(embeddings_path)
    metadata_path = locations.resolve_directory(metadata_path)
    embeddings_name = str(embeddings_path)
    metadata_name = str(metadata_path)
    if embeddings_name.endswith(".npz"):
        embeddings = load_embeddings_npz(embeddings_path)
    elif embeddings_name.endswith(".npy"):
        embeddings = np.asarray(np.load(embeddings_path), dtype=float)
    else:
        raise ValueError("embeddings must be .npz or .npy")

    if metadata_name.endswith(".jsonl"):
        metadata = load_metadata_jsonl(metadata_path)
    elif metadata_name.endswith(".csv"):
        metadata = load_metadata_csv(metadata_path)
    else:
        raise ValueError("metadata must be .jsonl or .csv")

    if len(metadata) != len(embeddings):
        raise ValueError("metadata length must match number of embeddings")
    return embeddings, metadata


def save_token_data(embeddings, metadata, directory=None, name=None):
    '''save token embeddings and metadata.
    embeddings              token embedding matrix
    metadata                token metadata records
    directory               optional storage directory
    name                    optional base name without extension
    '''
    directory = locations.resolve_directory(directory)
    name = locations.resolve_token_name(name)
    utils.ensure_directory(directory)
    embeddings_path = locations.token_embeddings_path(directory, name)
    metadata_path = locations.token_metadata_path(directory, name)
    np.savez(embeddings_path, embeddings=np.asarray(embeddings, dtype=float))
    save_metadata_jsonl(metadata, metadata_path)
    return {"embeddings": embeddings_path, "metadata": metadata_path}


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
    validate_prototype_metadata(metadata)
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
    validate_prototype_metadata(metadata)
    if len(vectors) != len(metadata):
        raise ValueError("metadata length must match number of vectors")
    return vectors, metadata


def load_embeddings_npz(path):
    '''load embeddings from npz.
    path                    npz file path
    '''
    data = np.load(path, allow_pickle=False)
    if "embeddings" not in data:
        raise ValueError("npz file must contain 'embeddings'")
    embeddings = np.asarray(data["embeddings"], dtype=float)
    return embeddings


def load_metadata_jsonl(path):
    '''load metadata rows from jsonl.
    path                    jsonl file path
    '''
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            utils.infer_label_key(row)
            rows.append(row)
    return rows


def load_metadata_csv(path):
    '''load metadata rows from csv.
    path                    csv file path
    '''
    with open(path) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        utils.infer_label_key(row)
    return rows


def resolve_token_paths(embeddings_path, metadata_path, directory, name):
    '''resolve token data paths.
    embeddings_path         optional embedding file path
    metadata_path           optional metadata file path
    directory               optional storage directory
    name                    optional base name without extension
    '''
    if embeddings_path is None:
        embeddings_path = locations.token_embeddings_path(directory, name)
    if metadata_path is None:
        metadata_path = locations.token_metadata_path(directory, name)
    return embeddings_path, metadata_path


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


def validate_prototype_metadata(rows):
    '''validate stored prototype metadata rows.
    rows                     prototype metadata records
    '''
    required = [
        "label",
        "unit_type",
        "source_phraser_keys",
        "source_echoframe_keys",
        "n_occurrences",
    ]
    for row in rows:
        for key in required:
            if key not in row:
                raise ValueError(f"prototype metadata row must contain '{key}'")
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
