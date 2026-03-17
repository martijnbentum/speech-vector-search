import csv
import json

import numpy as np

from speech_vector_search import utils


def load_token_data(embeddings_path, metadata_path):
    '''load token embeddings and metadata.
    embeddings_path         npz or npy file path
    metadata_path           jsonl or csv file path
    '''
    if embeddings_path.endswith(".npz"):
        embeddings = load_embeddings_npz(embeddings_path)
    elif embeddings_path.endswith(".npy"):
        embeddings = np.asarray(np.load(embeddings_path), dtype=float)
    else:
        raise ValueError("embeddings must be .npz or .npy")

    if metadata_path.endswith(".jsonl"):
        metadata = load_metadata_jsonl(metadata_path)
    elif metadata_path.endswith(".csv"):
        metadata = load_metadata_csv(metadata_path)
    else:
        raise ValueError("metadata must be .jsonl or .csv")

    if len(metadata) != len(embeddings):
        raise ValueError("metadata length must match number of embeddings")
    return embeddings, metadata


def save_metadata_jsonl(rows, path):
    '''save metadata rows to jsonl.
    rows                     metadata records
    path                     jsonl file path
    '''
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def save_prototypes(vectors, metadata, output_dir, config=None):
    '''save prototype vectors and metadata.
    vectors                  prototype matrix
    metadata                 prototype metadata records
    output_dir               directory to save files in
    config                   optional config dict to save
    '''
    vectors_path = output_dir + "/prototypes.npy"
    metadata_path = output_dir + "/metadata.jsonl"
    np.save(vectors_path, np.asarray(vectors, dtype=float))
    save_metadata_jsonl(metadata, metadata_path)
    if config is not None: utils.write_json(config, output_dir + "/config.json")
    return {
        "vectors": vectors_path,
        "metadata": metadata_path,
        "config": output_dir + "/config.json" if config is not None else None,
    }


def load_prototypes(vectors_path, metadata_path):
    '''load saved prototypes.
    vectors_path             npy file path
    metadata_path            jsonl file path
    '''
    vectors = np.asarray(np.load(vectors_path), dtype=float)
    metadata = load_metadata_jsonl(metadata_path)
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
