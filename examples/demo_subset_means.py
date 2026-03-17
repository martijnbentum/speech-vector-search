import json
import os
import tempfile

import numpy as np

from speech_vector_search.io import (
    load_token_data,
    save_metadata_jsonl,
    save_prototypes,
)
from speech_vector_search.prototypes import build_subset_mean_prototypes
from speech_vector_search.search import PrototypeIndex
from speech_vector_search.utils import ensure_directory


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
        ensure_directory(directory)
        np.savez(os.path.join(directory, "tokens.npz"), embeddings=embeddings)
        save_metadata_jsonl(metadata, os.path.join(directory, "tokens.jsonl"))

        loaded_embeddings, loaded_metadata = load_token_data(
            os.path.join(directory, "tokens.npz"),
            os.path.join(directory, "tokens.jsonl"),
        )
        vectors, rows, config = build_subset_mean_prototypes(
            loaded_embeddings,
            loaded_metadata,
            subset_size=3,
            n_subsets=1,
            min_count=3,
            seed=3,
        )
        save_prototypes(vectors, rows, directory, config=config)
        index = PrototypeIndex(vectors, rows)
        result = index.query_by_index(0, top_k=2)
        print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
