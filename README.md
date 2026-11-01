# Connection-disruption risk ranking: synthetic decision-support prototype (v0.3)

When a flight disruption leaves limited staff to help passengers, who should be helped first? This project ranks the passengers on 10,000 simulated disruption events by their risk of a severe connection disruption, with an event-level train/test split, and compares gradient boosting (HistGradientBoosting, HGB) with a simple rule and logistic regression; a separate comparison adds Random Forest, ExtraTrees and an MLP. With capacity to help the top 10% of the ranked queue, HGB finds 32.7% of the severe cases in the test events, against 22.0% for the simple rule and 10% for random selection; its lead over logistic regression is small. This is an educational prototype on simulated data: it is not a British Airways system, does not use British Airways operational data, and says nothing about performance on real disruptions.

## Results at a glance

**capture@k** is recall among the top k% of the ranked queue: the share of all severe cases in the test events that fall within the top floor(n × k) passengers. The test events hold 25,119 passengers and 2,654 severe cases; at 10% capacity the queue is 2,511 passengers.

| Ranking | capture@10% (95% interval) | Severe cases captured | Test ROC-AUC |
|---|---:|---:|---:|
| Random selection (expected) | 10.0% | about 265 | – |
| Reconstructed simple rule | 22.0% (20.6–23.7) | 585 | 0.7612 |
| Logistic regression, unweighted | 30.9% (29.3–32.5) | 819 | 0.8078 |
| HGB (saved refit queue) | 32.7% (31.1–34.3) | 868 | 0.8150 |

The simple rule ranks by `forecast_delay_minutes + expected_transfer_minutes − connection_buffer_minutes` (reconstructed; see [Limitations](#limitations)).

At 10% capacity, HGB captures about 3.3 times as many severe cases as random selection (868 against about 265).

In a separate comparison of five model families on the training events only, a decision rule fixed before the run selected the MLP on mean capture (capture@k averaged over 5, 10, 20 and 30%), but its lead over HGB in cross-validation, 0.0017 (0.17 pp), is smaller than the fold-to-fold spread. In a final check on the already-inspected test events, the MLP's lead was 0.0088 (0.88 pp, 95% interval 0.26–1.33 pp). HGB remains the primary model because it needed about 9 times less cross-validation fitting time (21 against 195 minutes) and the ranking results were produced with it; see [Model comparison](#model-comparison).

Sources: [`point_estimates_test.csv`](followup_outputs/point_estimates_test.csv), [`bootstrap_capture_ci.csv`](followup_outputs/bootstrap_capture_ci.csv).

## Problem and target

The target is `severe_connection_disruption = 1` when both conditions hold:

1. The original connection is missed.
2. No simulated recovery itinerary reaches the final destination within the same local calendar day as the originally scheduled final arrival.

This is the definition in the [frozen specification](data/synthetic_spec_v03_frozen.md), not a claim about British Airways' internal definitions. The meaning of the same-day flag in the saved data is not fully resolved; see [Limitations](#limitations).

## Data and evaluation design

- **Events and passengers:** 10,000 disruption events with 100,072 passenger itineraries, 5–15 per event.
- **Split:** event-level 75/25 split of the unique event IDs (`random_state=42`); no event appears in both sets. The membership is saved in `event_split/`.
- **Training set:** 7,500 events, 74,953 passenger rows, 10.3% severe.
- **Test set:** 2,500 events, 25,119 passenger rows, 2,654 severe cases (about 10.6%).
- **Features (10):** weather severity, ATC restriction, airport congestion, inbound operational pressure, forecast delay, connection buffer, expected transfer time, expected alternative flights, expected recovery options, and number of connections. These are simulator features, not validated live operational signals.
- **Test-set inspection:** I inspected the test events during development. They were held out from model fitting under the split above, but they are not an untouched final test set, so every test-set figure in this README, including the model comparison's final check, should be read as supporting evidence rather than an unbiased final estimate. The original v0.3 cross-validation also appears to have covered the test events ([docs/PROVENANCE.md](docs/PROVENANCE.md)), whereas the model comparison's cross-validation used training events only.

**Data origin.** The disruption events, operational features and outcomes are simulated. The nine booking-context columns (`num_passengers`, `trip_type`, `purchase_lead`, `length_of_stay`, `flight_hour`, `flight_day`, `booking_origin`, `route_origin`, `route_dest`) take their values from the Forage British Airways job-simulation booking dataset (`customer_booking.csv`, not included). How they were sampled was not recorded, and the models don't use these columns. The frozen specification says the Forage booking variables "are tested"; that test was planned but not done. The project started from the Forage British Airways data-science job simulation; the problem, target and data design here are a redesign.

## Results

### Capture under limited capacity

capture@k on the test events, with 95% intervals from an event-level bootstrap (2,000 resamples of the test events, conditional on the saved rankings; no refitting). Capacity is global: the top floor(25,119 × capacity) passengers are selected (1,255 / 2,511 / 5,023 / 7,535).

| Capacity | Random (expected) | Reconstructed simple rule | Logistic, balanced | Logistic, unweighted | HGB, saved refit queue |
|---|---:|---:|---:|---:|---:|
| 5% | 5.0% | 11.8% (10.6–13.1) | 17.4% (16.2–18.6) | 18.5% (17.2–19.8) | 19.4% (18.1–20.7) |
| 10% | 10.0% | 22.0% (20.6–23.7) | 30.1% (28.3–31.6) | 30.9% (29.3–32.5) | 32.7% (31.1–34.3) |
| 20% | 20.0% | 42.7% (40.8–44.6) | 51.1% (49.2–53.0) | 51.8% (50.0–53.8) | 53.9% (52.0–55.7) |
| 30% | 30.0% | 59.6% (57.8–61.7) | 69.1% (67.1–70.8) | 69.3% (67.4–71.2) | 69.5% (67.9–71.5) |

Paired differences, percentage points (point estimate; 95% interval):

| Capacity | HGB − simple rule | HGB − logistic, unweighted |
|---|---:|---:|
| 5% | +7.6 (5.9 to 9.0) | +0.9 (−0.3 to 2.0) |
| 10% | +10.7 (8.6 to 12.5) | +1.8 (0.5 to 3.1) |
| 20% | +11.2 (9.0 to 13.3) | +2.1 (0.5 to 3.4) |
| 30% | +9.9 (8.0 to 12.0) | +0.2 (−0.8 to 1.6) |

HGB's advantage over the simple rule is clear at every capacity (all four intervals exclude zero). Its advantage over unweighted logistic regression is small, and its interval includes zero at 5% and 30%.

Sources: [`bootstrap_capture_ci.csv`](followup_outputs/bootstrap_capture_ci.csv), [`bootstrap_paired_differences.csv`](followup_outputs/bootstrap_paired_differences.csv); counts at every capacity in [`point_estimates_test.csv`](followup_outputs/point_estimates_test.csv).

### Discrimination

| Model | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|
| HGB, saved refit queue | 0.8150 | 0.3218 |
| Logistic, balanced | 0.8074 | 0.2922 |
| Logistic, unweighted | 0.8078 | 0.3011 |
| Reconstructed simple rule | 0.7612 | 0.2204 |

The original v0.3 report's figures, including its 5-fold CV (which appears to have covered the test events), and how they compare with the saved refit and 12 seed refits, are in [docs/PROVENANCE.md](docs/PROVENANCE.md).

Source: [`point_estimates_test.csv`](followup_outputs/point_estimates_test.csv).

## Model comparison

This comparison asks whether HGB holds up against other model families under identical conditions. Its numbers come from different fits from the results above and should not be mixed with them.

### Design

- **Data:** training events only (7,500 events, 74,953 rows), from `event_split/`.
- **Models:** logistic regression (unweighted), Random Forest, HGB, ExtraTrees and an MLP, all on the same 10 features. Logistic regression and the MLP get a `StandardScaler` fitted inside each training fold. No class weights, so Brier scores are comparable. HGB's built-in early stopping is disabled, because its validation slice ignores event groups.
- **Evaluation:** event-grouped, stratified 5-fold cross-validation, repeated 3 times (15 outer folds).
- **Tuning:** nested; every model gets the same random-search budget (12 settings, event-grouped 3-fold inner validation), scored on the selection measure.
- **Decision rule:** the highest *mean capture* (the average of capture@5%, 10%, 20% and 30%) wins, even if the gap is small. The rule was fixed before the full comparison run: its text is recorded in [`run_config.json`](model_selection_outputs/run_config.json), which was written before the run's first fold result (see [docs/PROVENANCE.md](docs/PROVENANCE.md)).

### Cross-validation results

| Model | Mean capture (± sd) | capture@10% | ROC-AUC | PR-AUC | Brier (raw) | Total fit time (15 folds, incl. tuning) |
|---|---:|---:|---:|---:|---:|---:|
| MLP neural network | 0.4507 ± 0.0069 | 0.3352 | 0.8227 | 0.3242 | 0.0793 | 195 min |
| HistGradientBoosting | 0.4490 ± 0.0071 | 0.3371 | 0.8195 | 0.3201 | 0.0796 | 21 min |
| Random Forest | 0.4470 ± 0.0073 | 0.3343 | 0.8175 | 0.3200 | 0.0797 | 35 min |
| ExtraTrees | 0.4468 ± 0.0071 | 0.3350 | 0.8168 | 0.3179 | 0.0801 | 12 min |
| Logistic regression | 0.4376 ± 0.0061 | 0.3192 | 0.8108 | 0.3023 | 0.0810 | 1 min |

The `±` is the sample standard deviation across the 15 outer folds, not a standard error. Fit time is summed over all folds, including nested tuning, on my Windows machine.

Paired differences in mean capture, MLP minus each other model, over the same 15 folds:

| Compared with | Mean difference | Sd of differences | Folds where the MLP was ahead |
|---|---:|---:|---:|
| HistGradientBoosting | 0.0017 | 0.0032 | 10 of 15 |
| Random Forest | 0.0037 | 0.0022 | 14 of 15 |
| ExtraTrees | 0.0039 | 0.0032 | 12 of 15 |
| Logistic regression | 0.0131 | 0.0048 | 15 of 15 |

- The rule selects the MLP, but its lead over HGB (0.0017) is smaller than the fold-to-fold spread of the paired differences (0.0032). This is descriptive, not a formal test.
- The top four models are within about 0.004 of each other. Logistic regression is lowest, behind the MLP in all 15 folds.
- HGB has the highest capture@10% alone (0.3371 against 0.3352 for the MLP).
- Picking the best of five makes the winner's cross-validated score slightly optimistic.

Sources: [`summary_by_model.csv`](model_selection_outputs/summary_by_model.csv), [`paired_vs_winner.csv`](model_selection_outputs/paired_vs_winner.csv), [`fold_scores.csv`](model_selection_outputs/fold_scores.csv).

### Final check on the test events

The MLP and HGB were each tuned again with a larger budget (30 settings, event-grouped 5-fold inner validation) on all 7,500 training events, refit, and scored on the test events. `final_check_v03.py` refuses to score the test events again in an output folder that already contains `test_metrics.csv`.

| Model | Mean capture (95% interval) | ROC-AUC (95% interval) | PR-AUC | Brier (raw) | Captured at 10% |
|---|---:|---:|---:|---:|---:|
| MLP neural network | 0.4479 (0.4342–0.4619) | 0.8214 (0.8141–0.8283) | 0.3297 | 0.0813 | 884 |
| HGB (final-check refit) | 0.4391 (0.4266–0.4531) | 0.8157 (0.8081–0.8229) | 0.3245 | 0.0818 | 870 |

Intervals come from 1,000 event-level resamples, conditional on the fitted models; they do not include the uncertainty from choosing among five models or from the tuning budget. The paired difference in mean capture, MLP minus HGB, is 0.0088 (95% interval 0.0026–0.0133); the MLP was ahead in 99.9% of resamples. This HGB refit differs from the saved refit queue used in [Results](#results).

Sources: [`test_metrics.csv`](final_check_outputs/test_metrics.csv), [`test_bootstrap_intervals.csv`](final_check_outputs/test_bootstrap_intervals.csv), [`test_paired_difference.json`](final_check_outputs/test_paired_difference.json).

### Choice of primary model

HGB needed about 9 times less total cross-validation fitting time than the MLP (21 against 195 minutes, including nested tuning) and about 13 times less tuning time in the final check (3.2 against 41.7 minutes), for a small MLP lead. The ranking results in this README were also produced with HGB. Cheaper models exist: ExtraTrees (12 minutes) and logistic regression (1 minute), with mean capture 0.0022 and 0.0114 below HGB's. Keeping HGB is a practical choice made after seeing the results; it is not evidence that HGB is better.

## Robustness and leakage checks

- **Shuffled labels:** 40 HGB fits on shuffled training labels give a mean test ROC-AUC of 0.505 (SE 0.007), consistent with no leakage ([`shuffle_check_40.csv`](followup_outputs/shuffle_check_40.csv)).
- **Ablation and buffer dominance:** dropping `connection_buffer_minutes` lowers HGB test ROC-AUC from 0.8146 to 0.6834, and dropping both it and `expected_alternative_flights` gives 0.6731 ([`ablations_hgb.csv`](followup_outputs/ablations_hgb.csv)). Each figure is the mean of five HGB refits (seeds 0–4), not the saved refit queue, so the full-feature figure (0.8146) differs from the saved refit's 0.8150. `expected_alternative_flights` is the mean of the simulator's recovery-alternatives draw. The models mostly learn the simulator's rules, above all the connection buffer.
- **Seed spread:** across 12 HGB seed refits, severe cases captured at 10% capacity range from 847 to 877 (saved refit queue: 868) ([`hgb_seed_spread.csv`](followup_outputs/hgb_seed_spread.csv)).
- **Early stopping:** the follow-up HGB refits (seed spread, ablations, shuffled labels) use scikit-learn's default early stopping, which switches on above 10,000 training rows (here 74,953) and holds out a 10% validation slice drawn by row, not by event. Test events are excluded before fitting, so no test data leaks, but this differs from the model comparison, which disables early stopping.
- **Alternative target:** `sensitivity_alt_target_v03.py` tests one alternative severity target (missed connection and no recovery, or a recovery delay over 180 minutes), without intervals ([`sensitivity_alt_target.csv`](followup_outputs/sensitivity_alt_target.csv)). It assumes one of the two readings of the recovery-time fields described under [Limitations](#limitations).

## Limitations

- **Synthetic data and simulator rules:** the models mostly learn the simulator's rules, recovery availability is an influential simulator assumption, and the 5/10/20/30% capacity levels are illustrative, not staffing levels. The results support no causal conclusions and do not establish performance on real disruptions.
- **Test-set inspection:** see [Data and evaluation design](#data-and-evaluation-design).
- **Provenance:** the original v0.3 data generator, the original training code for logistic regression, Random Forest and HGB, their exact hyperparameters and preprocessing, the calibration-training code, the bootstrap code, the simple-rule implementation, some row-level predictions and the oracle probabilities were not recovered. The HGB ranking used here is a saved refit, not an exact reproduction of the originally reported model, and the follow-up and model-comparison scripts are new code, not recovered originals. Details are in [docs/PROVENANCE.md](docs/PROVENANCE.md) and [`SOURCE_CODE_RECOVERY_STATUS.md`](source_status/SOURCE_CODE_RECOVERY_STATUS.md).
- **Target semantics:** two readings of the recovery-time fields give different results for the same-day flag, and no document defines their units. The headline results use the stored `severe_connection_disruption` flag and do not depend on this. Details are in [docs/PROVENANCE.md](docs/PROVENANCE.md).
- **Connection-miss rule:** a rule inferred from the data disagrees with the stored `original_connection_missed` label on 316 of 100,072 rows (0.3158%). These are not confirmed labelling errors; `check_connection_miss_rule.py` reproduces the check, and details are in [docs/PROVENANCE.md](docs/PROVENANCE.md).
- **Reconstructed simple rule:** the score `forecast_delay_minutes + expected_transfer_minutes − connection_buffer_minutes` is not documented in the specification or source notes. It reproduces the saved counts (314 / 585 / 1,134 / 1,582 at 5/10/20/30% capacity) exactly, but matching aggregate counts does not prove that the original formula has been recovered.
- **Capacity policy:** capture uses a global passenger-capacity percentage. This does not establish how capacity should be allocated in live operations; an earlier per-event rule, described in [docs/PROVENANCE.md](docs/PROVENANCE.md), gives figures that are not comparable.
- **Not for operational use:** the prototype ignores real constraints such as seat contention, accessibility needs, passenger circumstances, legal rights, fairness and human decision-making. It should not be used to make real passenger decisions.
- **Untested model families:** XGBoost, LightGBM, k-nearest neighbours and support vector machines were not tested, and neither was any model using the booking-context columns.

## Reproduce

Use Python 3.13 (my runs used 3.13.7). From a terminal:

```bash
git clone https://github.com/Jeswin987/connection-disruption-risk-ranking.git
cd connection-disruption-risk-ranking
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
# source .venv/bin/activate        # Linux/macOS
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests take well under a minute and fit only one model (the HGB seed-0 refit described under Environment below). They check the dataset counts and target rule, that the saved event split is reproduced, the headline numbers (recomputed from the saved files where possible), the `MANIFEST.csv` checksums, the README links, and the follow-up scripts' output-folder guard.

Rerun the follow-up analyses into a new folder (`out_dir` is required and may not be the committed `followup_outputs/`):

```bash
python reproduce_followup_v03.py . rerun_outputs
python sensitivity_alt_target_v03.py . rerun_outputs
python shuffle_check_v03.py . rerun_outputs
python check_connection_miss_rule.py .
```

`check_connection_miss_rule.py` takes only the package folder and prints its results.

Rerun the model comparison (about 4.4 hours of fitting) and the final check (about 45 minutes of tuning) on my Windows machine:

```bash
python model_selection_v03.py . --out-dir rerun_model_selection
python final_check_v03.py . --selection-dir rerun_model_selection --out-dir rerun_final_check
```

Always use a new `--out-dir`. In the committed folder, `model_selection_v03.py` refuses to run if the recorded settings or library versions differ; otherwise it skips the fits but rewrites the summary files. `--quick` runs a small smoke test whose results are not meaningful.

**Environment.** The committed model-comparison runs record Python 3.13.7, NumPy 2.5.2, pandas 3.0.5 and scikit-learn 1.9.0 in their config files ([`run_config.json`](model_selection_outputs/run_config.json), [`final_check_config.json`](final_check_outputs/final_check_config.json)), matching the pins in `requirements.txt`. Under those pins, refitting HGB with seed 0 reproduces the committed seed-0 row of `hgb_seed_spread.csv` exactly; `tests/test_hgb_seed0_refit.py` checks this and is the only test that fits a model. Anything that refits HGB can vary with library version. `requirements.txt` also lists SciPy: `model_selection_v03.py` imports it directly, and `final_check_v03.py` uses it only indirectly.

## Repository map

```text
.
├── data/                      # Frozen synthetic datasets, specification, validation
├── event_split/               # Saved train/test event IDs
├── predictions/               # Saved test predictions and offline outcomes
├── results/                   # Original v0.3 results, older v0.1/v0.2 outputs, earlier follow-up notes
├── followup_outputs/          # Follow-up point estimates, bootstrap, sensitivity and checks
├── model_selection_outputs/   # Model comparison: cross-validation outputs
├── final_check_outputs/       # Model comparison: final check on the test events
├── audit_outputs/             # Data audit report and connection-miss exception summary
├── source_status/             # Source recovery status and documented fragments
├── environment/               # Export-environment snapshots and version records
├── docs/                      # PROVENANCE.md: provenance detail behind this README
├── tests/                     # Fast pytest suite (one HGB refit; otherwise no model fitting)
├── reproduce_followup_v03.py
├── sensitivity_alt_target_v03.py
├── shuffle_check_v03.py
├── check_connection_miss_rule.py
├── model_selection_v03.py     # Model comparison (cross-validation)
├── final_check_v03.py         # Model comparison (final check)
├── requirements.txt
├── requirements-dev.txt       # requirements.txt plus pytest
├── pytest.ini
├── .gitattributes             # Keeps LF line endings so checksums hold on Windows clones
├── MANIFEST.csv               # Checksums for the original v0.3 export only; not updated for later files
├── MANIFEST.json
└── README.md
```

The older v0.1/v0.2 outputs in `results/` are historical and should not be confused with the v0.3 results. The original v0.3 audit and explainability documents there predate the follow-up analyses and the model comparison, so their statements, such as "HGB is the best model tested", should be read together with the intervals and comparison in this README.
