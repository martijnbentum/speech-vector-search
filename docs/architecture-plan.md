# Architecture Plan

## Repository Roles

- `phraser` is the source of truth for speech metadata and object identity.
- `to_vector` extracts model outputs from audio.
- `echoframe` stores extracted model outputs and related store metadata.
- `speech-vector-search` builds, stores, loads, and searches retrieval-ready artifacts.

This repo should not replace `phraser`, `to_vector`, or `echoframe`.
This repo should own the retrieval-ready layer built on top of them.

## Core Direction

This repo should keep file IO, but at the retrieval-artifact level rather than
as a generic raw-embedding store.

The intended flow is:

`phraser` + `to_vector` + `echoframe` -> retrieval-ready lexicon/prototype artifacts -> search and evaluation

`echoframe` stores model outputs.
`speech-vector-search` stores search-ready artifacts derived from them.

## Current Recommendation

This repo should own:

- converting selected speech units into retrieval-ready prototypes
- storing prototype artifacts
- building subset-mean prototypes
- searching prototypes
- evaluating retrieval

This repo should not own:

- raw speech annotation metadata
- model inference
- general-purpose frame/output storage

## Canonical Ownership

Canonical source-of-truth ownership should remain upstream.

`phraser` should own:

- `phraser_key`
- `start`
- `end`
- `speaker`
- `audio` or filename
- the speech-unit type and label source

`echoframe` should own:

- store entry identity
- `model_name`
- `layer`
- `output_type`
- store tags

`speech-vector-search` should own:

- the retrieval artifact schema
- token/prototype construction rules
- pooling configuration when frame outputs are reduced here
- prototype/search/evaluation artifacts

This repo may still store denormalized snapshot metadata for convenience and
reproducibility, but those copied fields are not the source of truth.

## Prototype Artifact Design

The retrieval-ready stored artifact should be a prototype collection.

It should contain:

- vectors: one vector per prototype
- metadata: one row per prototype
- config: artifact-level provenance and settings

Required metadata should include:

- `unit_type`
- `label`
- `source_echoframe_keys`
- `n_occurrences`

When the artifact is derived from `echoframe`, it should also include source
store references.

Artifact-level config should include:

- `model_name`
- `layer`
- `output_type`
- `pooling`
- `prototype_method`
- selected tags

Allowed `prototype_method` values for now:

- `single_occurrence`
- `mean`

Optional copied metadata may include aggregated or copied source information:

- `start`
- `end`
- `speaker`
- `audio`

## Why a Translation Layer Is Still Needed

`echoframe` stores model outputs. This repo stores prototypes.

Those are not automatically the same thing. A prototype may be built from one
or more source occurrences.

The translation layer must define:

- which `phraser` objects are the source occurrences
- which `echoframe` entries correspond to those objects
- how stored frame outputs are reduced to one vector per source occurrence
- how one or more source occurrences are combined into one prototype
- what metadata is attached to each resulting prototype

Tags are useful for selecting relevant `echoframe` entries, but they do not
replace the semantic conversion from model outputs to retrieval units.

## Retrieval Units

The retrieval unit can be one of:

- `phone`
- `syllable`
- `word`
- `phrase`

The selected unit type should be explicit in stored metadata and config.
The stored searchable item is always a prototype, even when it is based on a
single source occurrence.

## Default Modeling Choices

- default output type: `hidden_state`
- default pooling: mean pooling over frames
- `phraser_key` is required because `phraser` is the source of truth for
  speech-related metadata

## Suggested Modules

- `speech_vector_search/store_ingest.py`
  Loads source vectors from `echoframe` and converts them into prototypes.

- `speech_vector_search/phraser_adapter.py`
  Resolves `phraser` objects, metadata, and unit selection.

- `speech_vector_search/pooling.py`
  Reduces frame outputs to one vector per source occurrence.

- `speech_vector_search/io.py`
  Persists prototype artifacts.

- `speech_vector_search/prototypes.py`
  Builds subset-mean prototypes from retrieval-ready artifacts.

- `speech_vector_search/search.py`
  Searches prototype vectors.

- `speech_vector_search/evaluate.py`
  Evaluates retrieval behavior.

## File IO Position

This repo should keep file IO, but not as the primary raw model-output storage
layer.

File IO should be used for:

- storing prototypes
- loading prototypes

This keeps the repo useful on its own while avoiding overlap with
`echoframe`.

## Pros

- clean separation of concerns across repositories
- reuse of `echoframe` for model-output storage
- explicit prototype artifacts for reproducibility
- file IO remains useful without duplicating upstream storage
- search stays fast and local once artifacts are materialized

## Cons

- one more materialized layer exists on disk
- a translation step is still required from stored outputs to prototypes
- the repo becomes more dependent on upstream semantics

## Recommended Next Steps

1. Define the prototype artifact schema.
2. Add `store_ingest.py`, `phraser_adapter.py`, and `pooling.py`.
3. Keep `io.py` prototype-only.
4. Keep `prototypes.py`, `search.py`, and `evaluate.py` storage-agnostic.
5. Add tests around store ingestion and artifact creation.
6. Add one end-to-end example from `phraser` + `echoframe` into this repo.
