"""Refitting HGB with seed 0 reproduces the committed seed-spread row.

This is the one test that fits a model. It checks that the installed library
versions reproduce the committed follow-up tables.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from conftest import ROOT
from model_selection_v03 import CAPACITIES, FEATURES, TARGET_COL

SEED_SPREAD_CSV = ROOT / "followup_outputs" / "hgb_seed_spread.csv"


def test_hgb_seed0_refit_matches_committed_row(train_rows, test_rows):
    # Same fit, scoring and capture as the `for s in range(12)` seed-spread loop
    # in reproduce_followup_v03.py:
    # `HistGradientBoostingClassifier(random_state=s).fit(tr[FEATURES], tr.severe_connection_disruption)`
    # with its FEATURES list (identical, in the same order, to model_selection_v03.FEATURES),
    # `np.argsort(-p, kind="mergesort")` and point_capture()'s `int(np.floor(n * cap))`.
    model = HistGradientBoostingClassifier(random_state=0).fit(
        train_rows[FEATURES], train_rows[TARGET_COL]
    )
    p = model.predict_proba(test_rows[FEATURES])[:, 1]
    y = test_rows[TARGET_COL].to_numpy()
    order = np.argsort(-p, kind="mergesort")
    captured = [int(y[order[: int(np.floor(len(y) * c))]].sum()) for c in CAPACITIES]

    saved = pd.read_csv(SEED_SPREAD_CSV).set_index("seed").loc[0]
    assert roc_auc_score(y, p) == saved["roc_auc"]
    assert captured == [int(saved[f"captured@{int(c * 100)}%"]) for c in CAPACITIES]
