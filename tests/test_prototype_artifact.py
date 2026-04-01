from speech_vector_search import prototype_artifact


def test_make_prototype_row_sets_occurrence_count():
    row = prototype_artifact.make_prototype_row("a", "word", ["p0", "p1"],
        ["e0", "e1"], subset_id=3)
    assert row["n_occurrences"] == 2
    assert row["subset_id"] == 3


def test_validate_rows_rejects_mismatched_source_lengths():
    rows = [
        {
            "label": "a",
            "unit_type": "word",
            "source_phraser_keys": ["p0"],
            "source_echoframe_keys": ["e0", "e1"],
            "n_occurrences": 1,
        }
    ]
    try:
        prototype_artifact.validate_rows(rows)
    except ValueError as exc:
        assert "must match in length" in str(exc)
        return
    raise AssertionError("expected ValueError for mismatched source lengths")
