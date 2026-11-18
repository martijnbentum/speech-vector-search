# speech-vector-search

Small Python package for vector search over speech-model prototypes.

## What it does

The package works with prototype artifacts on disk. It:

- stores prototype vectors and metadata
- loads prototype vectors and metadata
- searches prototypes with FAISS (brute-force numpy fallback available)
- evaluates simple same-word retrieval metrics

## Installation

Editable install:

```bash
uv pip install -e .
```

With tests:

```bash
uv pip install -e .[test]
```

From git:

```bash
uv pip install git+https://github.com/martijnbentum/speech-vector-search.git
```

Optional extras from git:

```bash
uv pip install 'speech-vector-search[test] @ \
git+https://github.com/martijnbentum/speech-vector-search.git'
```

## Quick example

```python
import numpy as np

from speech_vector_search import io
from speech_vector_search import prototypes
from speech_vector_search.search import PrototypeIndex

vectors = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=float)
token_rows = [
    {'label': 'hello', 'unit_type': 'word', 'echoframe_key': 'e0'},
    {'label': 'hello', 'unit_type': 'word', 'echoframe_key': 'e1'},
]

prototype_vectors, prototype_rows, _ = prototypes.build_subset_prototypes(
    'hello', vectors, token_rows, subset_size=2, n_subsets=1)
io.save_prototypes(prototype_vectors, prototype_rows, name='word_demo')

loaded_vectors, loaded_rows = io.load_prototypes(name='word_demo')
result = PrototypeIndex(loaded_vectors, loaded_rows).query_by_index(0, top_k=1)
print(result['scores'])
print(result['metadata'][0].label)
```

## Batched scores and exact ranks

`VectorIndex.cosine_batches()` yields `(start, similarities)` batches against
all indexed candidates. Use separate ranking functions on the same score row
when you need both neighbours and ranks for specified candidates:

```python
from speech_vector_search import ranking

for start, scores in index.cosine_batches(query_vectors, batch_size=32):
    for offset, similarities in enumerate(scores):
        query_index = query_indices[start + offset]
        neighbours = ranking.k_nearest_neighbours(similarities, top_k=30,
            excluded_index=query_index)
        ranks = ranking.global_ranks(similarities, candidate_indices)
        neighbour_scores = similarities[neighbours]
        candidate_scores = similarities[candidate_indices]
```

`query_indices` identifies which indexed token to exclude for each external
query. Omit `excluded_index` when no candidate should be excluded. Candidate
indices to rank are supplied by the caller and may differ for each query.
Ranks include every indexed candidate, regardless of neighbour exclusion.
They are zero-based, with higher similarities first and exact ties ordered by
ascending candidate index. Rank calculations retain the scoring precision:
float32 for the FAISS backend and float64 for the brute-force backend.

This exhaustive scoring path uses NumPy matrix multiplication once per batch;
it does not additionally call FAISS search. Temporary score matrices contain
at most `batch_size` rows. Existing `query()` and `queries()` remain available
for nearest-neighbour searches through the selected backend.

Candidate vectors are validated and prepared once when the index is built.
Cosine batches reuse the backend's candidate matrix, including the retained
float32 matrix used to build FAISS. Treat indexed vectors as fixed; rebuild
the index to change them.

## Notes

- FAISS is a required dependency. The package also supports a brute-force numpy backend via the `backend="brute_force"` option.
- Prototype vectors are L2-normalized, so cosine similarity is computed with dot products.
- Metadata rows stay aligned with vectors during save, load, search, and evaluation.
- The core public surface is `io`, `metadata`, `prototypes`, `search`, and `evaluate`.
- The current prototype builder returns `prototype_method='subset_mean'` configs.
- Git tag `pre_echoframe` marks the repository state before echoframe-related changes.
