import json
import os
import tempfile

import numpy as np

from speech_vector_search import io
from speech_vector_search import prototypes
from speech_vector_search import search
from speech_vector_search import utils


def main():
    '''run small subset-means demo.
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

    with tempfile.TemporaryDirectory() as directory:
        utils.ensure_directory(directory)
        np.savez(os.path.join(directory, "tokens.npz"), embeddings=embeddings)
        metadata_path = os.path.join(directory, "tokens.jsonl")
        io.save_metadata_jsonl(metadata, metadata_path)

        loaded_embeddings, loaded_metadata = io.load_token_data(
            os.path.join(directory, "tokens.npz"),
            os.path.join(directory, "tokens.jsonl"),
        )
        vectors, rows, config = prototypes.build_subset_mean_prototypes(
            loaded_embeddings, loaded_metadata, subset_size=3, n_subsets=1,
            min_count=3, seed=3)
        io.save_prototypes(vectors, rows, directory, config=config)
        index = search.PrototypeIndex(vectors, rows)
        result = index.query_by_index(0, top_k=2)
        print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
