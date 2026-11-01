"""Headline numbers reported in README.md, recomputed from saved files where possible."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from conftest import ROOT
from model_selection_v03 import CAPACITIES, TARGET_COL, capture_profile

SEVERE_IN_TEST = 2_654
QUEUE_CSV = ROOT / "results" / "ba_disruption_decision_support_v03__priority_queue_evaluation_v03.csv"
POINT_ESTIMATES_CSV = ROOT / "followup_outputs" / "point_estimates_test.csv"
SUMMARY_BY_MODEL_CSV = ROOT / "model_selection_outputs" / "summary_by_model.csv"
FINAL_PREDICTIONS_CSV = ROOT / "final_check_outputs" / "test_predictions.csv"
FINAL_METRICS_CSV = ROOT / "final_check_outputs" / "test_metrics.csv"


def floor_capacity(n, capacity):
    # Floor rule from `int(np.floor(n * cap))` in point_capture() in reproduce_followup_v03.py.
    return int(np.floor(n * capacity))


def point_capture(y, scores, capacity):
    # Ranking and capture from `np.argsort(-v, kind="mergesort")` and point_capture()
    # in reproduce_followup_v03.py
    # (descending score, stable mergesort, top floor(n * capacity)).
    order = np.argsort(-scores, kind="mergesort")
    return int(y[order[:floor_capacity(len(y), capacity)]].sum())


@pytest.fixture(scope="module")
def saved_hgb_scores(test_rows):
    # Join of the saved queue's risk_score, from the `q[["passenger_id", "risk_score"]]`
    # merge in reproduce_followup_v03.py.
    queue = pd.read_csv(QUEUE_CSV)
    scores = test_rows[["passenger_id"]].merge(
        queue[["passenger_id", "risk_score"]], on="passenger_id", how="left"
    )["risk_score"].to_numpy()
    assert not np.isnan(scores).any()
    return scores


def test_global_capacity_floors(test_rows):
    assert [floor_capacity(len(test_rows), c) for c in CAPACITIES] == [1_255, 2_511, 5_023, 7_535]


def test_saved_hgb_queue_capture(test_rows, saved_hgb_scores):
    y = test_rows[TARGET_COL].to_numpy()
    assert [point_capture(y, saved_hgb_scores, c) for c in CAPACITIES] == [516, 868, 1_431, 1_845]


def test_saved_hgb_queue_roc_auc(test_rows, saved_hgb_scores):
    auc = roc_auc_score(test_rows[TARGET_COL], saved_hgb_scores)
    assert round(auc, 4) == 0.8150


def test_simple_rule_baseline_capture(test_rows):
    # Reconstructed rule from `rule = (te.forecast_delay_minutes + ...` in reproduce_followup_v03.py.
    rule = (
        test_rows.forecast_delay_minutes
        + test_rows.expected_transfer_minutes
        - test_rows.connection_buffer_minutes
    ).to_numpy()
    y = test_rows[TARGET_COL].to_numpy()
    assert [point_capture(y, rule, c) for c in CAPACITIES] == [314, 585, 1_134, 1_582]


def test_logistic_capture_at_10pct():
    # Read from the saved file: recomputing would mean fitting the models.
    est = pd.read_csv(POINT_ESTIMATES_CSV).set_index("strategy")["captured@10%"]
    assert est["Logistic_unweighted"] == 819
    assert est["Logistic_balanced"] == 799


def test_model_comparison_mean_capture():
    summary = pd.read_csv(SUMMARY_BY_MODEL_CSV).set_index("model")["mean_capture_mean"]
    assert round(summary["mlp_neural_network"], 4) == 0.4507
    assert round(summary["hist_gradient_boosting"], 4) == 0.4490


@pytest.mark.parametrize(
    "model, expected_count",
    [("mlp_neural_network", 884), ("hist_gradient_boosting", 870)],
)
def test_final_check_top10_counts(model, expected_count):
    # Recomputed from the saved predictions with the floor rule and tie-break
    # used by final_check_v03.py (model_selection_v03.capture_profile).
    preds = pd.read_csv(FINAL_PREDICTIONS_CSV)
    y = preds[TARGET_COL].to_numpy()
    assert len(preds) == 25_119 and int(y.sum()) == SEVERE_IN_TEST

    share = capture_profile(y, preds[f"score_{model}"].to_numpy())[10]
    assert share * SEVERE_IN_TEST == pytest.approx(expected_count, abs=1e-6)

    saved = pd.read_csv(FINAL_METRICS_CSV).set_index("model").loc[model, "capture_10pct"]
    assert share == pytest.approx(saved, abs=1e-12)
