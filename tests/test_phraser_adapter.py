from speech_vector_search import metadata


def test_resolve_unit_type_alias():
    assert metadata.resolve_unit_type("phoneme") == "phone"
    assert metadata.resolve_unit_type({"unit_type": "phrases"}) == "phrase"


def test_resolve_unit_type_from_object_attribute():
    class Dummy:
        unit_type = "words"

    assert metadata.resolve_unit_type(Dummy()) == "word"


def test_resolve_shared_unit_type_rejects_mismatch():
    rows = [{"unit_type": "word"}, {"unit_type": "phone"}]
    try:
        metadata.resolve_shared_unit_type(rows)
    except ValueError as exc:
        assert "share one unit_type" in str(exc)
        return
    raise AssertionError("expected ValueError for mismatched unit types")
