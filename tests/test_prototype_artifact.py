from speech_vector_search import prototypes


def test_make_prototype_row_sets_occurrence_count():
    row = prototypes.make_prototype_row('a', 3, [
        {
            'unit_type': 'word',
            'phraser_key': 'p0',
            'echoframe_key': 'e0',
        },
        {
            'unit_type': 'word',
            'phraser_key': 'p1',
            'echoframe_key': 'e1',
        },
    ])
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
        prototypes.validate_rows(rows)
    except ValueError as exc:
        assert "must match in length" in str(exc)
        return
    raise AssertionError("expected ValueError for mismatched source lengths")
