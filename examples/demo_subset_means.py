import json
import tempfile

import numpy as np

from speech_vector_search import io
from speech_vector_search import prototypes
from speech_vector_search import search
from speech_vector_search import utils


def main():
    '''run small subset-means demo.
    '''
    embeddings, metadata = make_demo_tokens()
    with tempfile.TemporaryDirectory() as directory:
        utils.ensure_directory(directory)
        embeddings_path, metadata_path = save_demo_tokens(directory, embeddings,
            metadata)
        loaded_embeddings, loaded_metadata = load_demo_tokens(embeddings_path,
            metadata_path)
        vectors, rows = build_demo_prototypes(directory, loaded_embeddings,
            loaded_metadata)
        show_demo_query(vectors, rows)


def make_demo_tokens():
    '''create small token embeddings and metadata.
    no parameters            returns demo embeddings and metadata
    '''
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.0, 1.0],
        ]
    )
    metadata = [
        {"word": "hello", "id": "h1", "speaker": "spk1"},
        {"word": "hello", "id": "h2", "speaker": "spk2"},
        {"word": "hello", "id": "h3", "speaker": "spk1"},
        {"word": "world", "id": "w1", "speaker": "spk1"},
        {"word": "world", "id": "w2", "speaker": "spk2"},
        {"word": "world", "id": "w3", "speaker": "spk2"},
    ]
    return embeddings, metadata


def save_demo_tokens(directory, embeddings, metadata):
    '''save demo token data to a temporary directory.
    directory                directory for temporary files
    embeddings               token embedding matrix
    metadata                 token metadata rows
    '''
    paths = io.save_token_data(embeddings, metadata, directory=directory)
    return paths["embeddings"], paths["metadata"]


def load_demo_tokens(embeddings_path, metadata_path):
    '''load demo token data from disk.
    embeddings_path          npz file path
    metadata_path            jsonl file path
    '''
    return io.load_token_data(embeddings_path, metadata_path)


def build_demo_prototypes( embeddings, metadata, save = False, overwrite = False,
    directory = None,  name = "prototypes"):
    '''build and save demo prototypes.
    embeddings               token embedding matrix
    metadata                 token metadata rows
    save                     whether to save prototypes to disk
    overwrite                whether to overwrite existing prototype files
    directory                directory for saved prototype files
    name                     base name for saved prototype files
    '''
    vectors, rows, config = prototypes.build_subset_mean_prototypes(
        embeddings, metadata, subset_size=3, n_subsets=1, min_count=3, seed=3)
    if save: 
        io.save_prototypes(vectors, rows, directory=directory,
            name="prototypes", config=config, overwrite=overwrite)
    return vectors, rows


def show_demo_query(vectors, rows):
    '''query the demo prototypes and print metadata.
    vectors                  prototype matrix
    rows                     prototype metadata rows
    '''
    index = search.PrototypeIndex(vectors, rows)
    result = index.query_by_index(0, top_k=2)
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
