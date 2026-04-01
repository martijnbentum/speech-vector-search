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
pip install -e .
```

With tests:

```bash
pip install -e .[test]
```

With optional FAISS:

```bash
pip install -e .[faiss]
```

From git:

```bash
uv pip install git+ssh://git@github.com/martijnbentum/speech-vector-search.git
```

## Quick example

```python
from speech_vector_search.io import load_prototypes
from speech_vector_search.search import PrototypeIndex

vectors, rows = load_prototypes(directory="data", name="word_demo")
index = PrototypeIndex(vectors, rows)
result = index.query_by_index(0, top_k=5)
print(result["scores"])
print(result["metadata"][0])
```

## Notes

- FAISS is optional. If `faiss` is not installed, the package falls back to brute-force cosine search with numpy.
- Prototype vectors are L2-normalized, so cosine similarity is computed with dot products.
- Metadata rows stay aligned with vectors during save, load, search, and evaluation.
- Allowed `prototype_method` values are currently `single_occurrence` and `mean`.
- Git tag `pre_echoframe` marks the repository state before echoframe-related changes.
