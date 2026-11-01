"""Shared fixtures for the fast artifact checks. No models are fitted."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Lets the tests import model_selection_v03, which has no module-level side
# effects (constants and function definitions only; main() is guarded).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASSENGERS_CSV = ROOT / "data" / "passenger_itineraries_v03_10000.csv"
TEST_IDS_CSV = ROOT / "event_split" / "test_event_ids.csv"
TRAIN_IDS_CSV = ROOT / "event_split" / "train_event_ids.csv"


@pytest.fixture(scope="session")
def passengers() -> pd.DataFrame:
    return pd.read_csv(PASSENGERS_CSV)


@pytest.fixture(scope="session")
def test_event_ids() -> pd.Series:
    return pd.read_csv(TEST_IDS_CSV)["event_id"]


@pytest.fixture(scope="session")
def train_event_ids() -> pd.Series:
    return pd.read_csv(TRAIN_IDS_CSV)["event_id"]


@pytest.fixture(scope="session")
def test_rows(passengers, test_event_ids) -> pd.DataFrame:
    # Same row selection as `te = P[P.event_id.isin(test_ids)]` in reproduce_followup_v03.py.
    return passengers[passengers.event_id.isin(set(test_event_ids))].reset_index(drop=True)


@pytest.fixture(scope="session")
def train_rows(passengers, test_event_ids) -> pd.DataFrame:
    # Same row selection as `tr = P[~P.event_id.isin(test_ids)]` in reproduce_followup_v03.py.
    return passengers[~passengers.event_id.isin(set(test_event_ids))].reset_index(drop=True)
