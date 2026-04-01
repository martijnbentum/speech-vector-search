import json

import numpy as np

from speech_vector_search import prototypes
from speech_vector_search import search


def main():
    '''run small subset-means demo.
    '''
    embeddings, metadata = make_demo_tokens()
    vectors, rows = build_demo_prototypes(embeddings, metadata)
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
        {"label": "hello", "id": "h1"},
        {"label": "hello", "id": "h2"},
        {"label": "hello", "id": "h3"},
        {"label": "world", "id": "w1"},
        {"label": "world", "id": "w2"},
        {"label": "world", "id": "w3"},
    ]
    return embeddings, metadata

def build_demo_prototypes(embeddings, metadata):
    '''build and save demo prototypes.
    embeddings               token embedding matrix
    metadata                 token metadata rows
    '''
    vectors, rows, config = prototypes.build_subset_mean_prototypes(
        embeddings, metadata, subset_size=3, n_subsets=1, min_count=3, seed=3)
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
