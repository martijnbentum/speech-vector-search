# speech-vector-search

Small Python package for vector search over speech-model prototypes.

## What it does

The package works with prototype artifacts on disk. It:

- stores prototype vectors and metadata
- loads prototype vectors and metadata
- searches prototypes with either numpy or optional FAISS
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

With optional FAISS:

```bash
uv pip install -e .[faiss]
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

## Notes

- FAISS is optional. If `faiss` is not installed, the package falls back to brute-force cosine search with numpy.
- Prototype vectors are L2-normalized, so cosine similarity is computed with dot products.
- Metadata rows stay aligned with vectors during save, load, search, and evaluation.
- The core public surface is `io`, `metadata`, `prototypes`, `search`, and `evaluate`.
- The current prototype builder returns `prototype_method='subset_mean'` configs.
- Git tag `pre_echoframe` marks the repository state before echoframe-related changes.
