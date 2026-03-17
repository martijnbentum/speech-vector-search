# speech-vector-search

Small Python package for vector search over speech-model word embeddings using subset means.

## What it does

The package assumes token-level embeddings already exist on disk. It:

- loads token embeddings and metadata
- groups tokens by word label
- samples deterministic non-overlapping subsets
- computes L2-normalized subset-mean prototypes
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

## Expected input format

First version expects:

- embeddings in `.npz`
- metadata in `.jsonl`

The `.npz` file should contain an array named `embeddings` with shape `(n_tokens, dim)`.

Each metadata row should at least contain:

```json
{"word": "hello", "id": "utt1_003"}
```

`label` can be used instead of `word`. Extra fields are preserved.

## Quick example

```python
from speech_vector_search.io import load_token_data
from speech_vector_search.prototypes import build_subset_mean_prototypes
from speech_vector_search.search import PrototypeIndex

embeddings, metadata = load_token_data("tokens.npz", "tokens.jsonl")
vectors, rows, config = build_subset_mean_prototypes(
    embeddings,
    metadata,
    subset_size=4,
    n_subsets=3,
    min_count=12,
    seed=7,
)

index = PrototypeIndex(vectors, rows)
result = index.query_by_index(0, top_k=5)
print(result["scores"])
print(result["metadata"][0])
```

## CLI

Build prototypes:

```bash
speech-vector-search build-prototypes \
  --embeddings tokens.npz \
  --metadata tokens.jsonl \
  --output-dir data/prototypes \
  --subset-size 4 \
  --n-subsets 3 \
  --min-count 12
```

Build index files:

```bash
speech-vector-search build-index \
  --vectors data/prototypes/prototypes.npy \
  --metadata data/prototypes/metadata.jsonl \
  --output-dir data/index
```

Query:

```bash
speech-vector-search query \
  --vectors data/index/prototypes.npy \
  --metadata data/index/metadata.jsonl \
  --query-index 0 \
  --top-k 5
```

Evaluate:

```bash
speech-vector-search evaluate \
  --vectors data/prototypes/prototypes.npy \
  --metadata data/prototypes/metadata.jsonl \
  --top-k 5
```

## Notes

- FAISS is optional. If `faiss` is not installed, the package falls back to brute-force cosine search with numpy.
- Prototype vectors are L2-normalized, so cosine similarity is computed with dot products.
- Metadata rows stay aligned with vectors during save, load, search, and evaluation.
