import numpy as np

from speech_vector_search import io
from speech_vector_search import prototypes
from speech_vector_search import search


source_vectors = np.array(
    [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ],
    dtype=float,
)

source_metadata = [
    {
        "label": "hello",
        "unit_type": "word",
        "phraser_key": "phr_word_0001",
        "echoframe_key": "echo_0001",
    },
    {
        "label": "hello",
        "unit_type": "word",
        "phraser_key": "phr_word_0002",
        "echoframe_key": "echo_0002",
    },
    {
        "label": "world",
        "unit_type": "word",
        "phraser_key": "phr_word_0003",
        "echoframe_key": "echo_0003",
    },
    {
        "label": "world",
        "unit_type": "word",
        "phraser_key": "phr_word_0004",
        "echoframe_key": "echo_0004",
    },
]

prototype_vectors, prototype_rows, config = (
    prototypes.build_subset_mean_prototypes(
        source_vectors,
        source_metadata,
        subset_size=2,
        n_subsets=1,
        min_count=2,
        seed=7,
    )
)

config.update(
    {
        "model_name": "wav2vec2-base",
        "layer": 7,
        "output_type": "hidden_state",
        "pooling": "mean",
        "prototype_method": "mean",
        "tags": ["demo", "dummy"],
    }
)

io.save_prototypes(
    prototype_vectors,
    prototype_rows,
    directory="data",
    name="dummy_prototype_lexicon",
    config=config,
)

loaded_vectors, loaded_rows = io.load_prototypes(
    directory="data",
    name="dummy_prototype_lexicon",
)

index = search.PrototypeIndex(loaded_vectors, loaded_rows, backend="brute_force")
result = index.query_by_index(0, top_k=2)

print(loaded_rows)
print(result["indices"])
print(result["metadata"])
