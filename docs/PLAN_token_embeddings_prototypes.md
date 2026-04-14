## Feature 1: Rename Prototype Metadata Builder

Status: implemented.

### Requirements

- Rename `make_prototype_row(...)` to `make_prototype_metadata(...)`.
- Update direct call sites, tests, and docs to use the new name.
- Keep behavior unchanged apart from the rename.
- This feature should not introduce the `TokenEmbeddings` adapter yet.

### Tests

- Existing prototype metadata tests pass with the new function name.
- Existing prototype-building tests pass after the rename.


## Feature 2: Add `TokenEmbeddings` Prototype Adapters

### Requirements

- Add a narrow adapter module:
  - `speech_vector_search/echoframe_adapter.py`
- Add:
  - `build_mean_prototype_from_token_embeddings(...)`
  - `build_subset_prototypes_from_token_embeddings(...)`
- Keep `speech_vector_search/prototypes.py` numpy-based.
- Adapter helpers accept only `echoframe.TokenEmbeddings`.
- Adapter helpers require explicit:
  - `label`
  - `unit_type`
- Adapter helpers reject:
  - token embeddings with `'frames'` in `dims`
  - token embeddings with more than one layer
- If the token embeddings contain exactly one layer, the adapter may collapse
  that layer internally.
- Adapter helpers should build token-level source rows from
  `token_embeddings.echoframe_keys` and then call the existing numpy-based
  prototype builders.
- Token arrays must have identical shapes; if `TokenEmbeddings.to_numpy()`
  cannot stack them, the adapter should raise rather than pad or pool.
- Do not re-export the adapter helpers from `speech_vector_search.__init__`.

### Tests

- `build_mean_prototype_from_token_embeddings(...)`
  - works for frame-aggregated token embeddings
  - returns the same vector and metadata content as the equivalent
    numpy-based path
  - uses `source_echoframe_keys` from the token embeddings
- `build_subset_prototypes_from_token_embeddings(...)`
  - returns the same vectors/rows/config as the equivalent numpy-based path
- Adapter rejects:
  - embeddings with `'frames'` in `dims`
  - embeddings with more than one layer
- Adapter accepts an exactly one-layer token-embedding input if the layer can
  be collapsed internally.
- Adapter raises when token arrays do not share one shape.
