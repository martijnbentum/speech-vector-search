import speech_vector_search


def test_package_exports_current_public_api():
    assert speech_vector_search.VectorIndex is speech_vector_search.search.VectorIndex
    assert (
        speech_vector_search.VectorMetadata
        is speech_vector_search.metadata.VectorMetadata
    )
    assert (
        speech_vector_search.VectorMetadatas
        is speech_vector_search.metadata.VectorMetadatas
    )
    assert speech_vector_search.__all__ == [
        'VectorIndex',
        'VectorMetadata',
        'VectorMetadatas',
        'metadata',
        'search',
        'util_math',
    ]
