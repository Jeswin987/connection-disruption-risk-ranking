"""Dataset shape, target definition and the saved event-level split."""
from sklearn.model_selection import train_test_split

from model_selection_v03 import GROUP_COL, TARGET_COL


def test_dataset_size(passengers):
    assert len(passengers) == 100_072
    assert passengers[GROUP_COL].nunique() == 10_000


def test_rows_per_event_between_5_and_15(passengers):
    sizes = passengers.groupby(GROUP_COL).size()
    assert sizes.min() >= 5
    assert sizes.max() <= 15


def test_target_is_missed_and_not_recovered_same_day(passengers):
    # Rule from reproduce_followup_v03.py:44.
    expected = (
        (passengers.original_connection_missed == 1)
        & (passengers.same_day_recovery_available == 0)
    ).astype(int)
    assert (expected == passengers[TARGET_COL]).all()


def test_saved_test_split_is_reproduced(passengers, test_event_ids):
    # Split procedure from reproduce_followup_v03.py:40-41.
    events = passengers[GROUP_COL].drop_duplicates()
    _, rebuilt_test = train_test_split(events, test_size=0.25, random_state=42)
    assert set(rebuilt_test) == set(test_event_ids)
    assert len(rebuilt_test) == len(test_event_ids)


def test_no_event_in_both_train_and_test(test_event_ids, train_event_ids):
    assert not set(test_event_ids) & set(train_event_ids)
    assert test_event_ids.is_unique and train_event_ids.is_unique


def test_test_split_counts(test_rows, test_event_ids):
    assert len(test_event_ids) == 2_500
    assert test_rows[GROUP_COL].nunique() == 2_500
    assert len(test_rows) == 25_119
    assert int(test_rows[TARGET_COL].sum()) == 2_654


def test_train_split_counts(train_rows, train_event_ids):
    assert len(train_event_ids) == 7_500
    assert set(train_rows[GROUP_COL]) == set(train_event_ids)
    assert len(train_rows) == 74_953
