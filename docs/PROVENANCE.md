# Provenance

This document holds the provenance detail behind the [README](../README.md): what was and wasn't recovered from the original v0.3 work, how the saved figures relate to each other, and the evidence behind some of the README's statements. Paths are relative to the repository root.

## What was not recovered

The following were not recovered, as listed in [`source_status/SOURCE_CODE_RECOVERY_STATUS.md`](../source_status/SOURCE_CODE_RECOVERY_STATUS.md):

- the original v0.3 synthetic data-generation source (the frozen specification documents the mechanism, but not the generator code);
- the original training source for logistic regression, Random Forest and HGB;
- the exact HGB and Random Forest hyperparameters, and the exact class-weight or resampling configuration;
- the original calibration-training source (the saved calibration tables exist);
- the original simple-rule implementation;
- row-level test predictions for logistic regression, Random Forest and the simple rule;
- the bootstrap source for the original HGB/simple-rule intervals;
- the oracle/latent-probability generator, or saved latent probabilities.

The original logistic preprocessing is also undocumented: the only documented setting is `class_weight="balanced"`, and the follow-up scripts add a `StandardScaler`. [`source_status/DOCUMENTED_CODE_FRAGMENTS.md`](../source_status/DOCUMENTED_CODE_FRAGMENTS.md) transcribes the documented decisions (event split, logistic configuration, per-event capacity rule); these are not original source.

The follow-up scripts (`reproduce_followup_v03.py`, `sensitivity_alt_target_v03.py`, `shuffle_check_v03.py`, `check_connection_miss_rule.py`) and the model-comparison scripts (`model_selection_v03.py`, `final_check_v03.py`) are new code. They can be rerun on the saved data, but rerunning them does not recover the original pipeline.

## Original HGB report, saved refit and seed refits

Severe cases captured on the test events (of 2,654):

| | 5% | 10% | 20% | 30% |
|---|---:|---:|---:|---:|
| Original HGB report | 518 | 863 | 1,432 | 1,862 |
| Saved HGB refit queue | 516 | 868 | 1,431 | 1,845 |
| 12 seed refits, mean ± sd (range) | 514.0 ± 6.6 (503–524) | 863.9 ± 7.9 (847–877) | 1,422.5 ± 8.5 (1,407–1,433) | 1,851.2 ± 5.5 (1,841–1,861) |

Discrimination reported by the original v0.3 work:

| Model | Test ROC-AUC | Test PR-AUC | 5-fold CV ROC-AUC |
|---|---:|---:|---:|
| HGB, original v0.3 report | 0.8162 | 0.3257 | 0.8193 ± 0.0026 |
| Random Forest, original v0.3 report | 0.8149 | 0.3213 | 0.8167 ± 0.0027 |

Random Forest has no capture counts because its row-level predictions were not recovered.

The original HGB hyperparameters, seed and training source were not recovered, so the saved refit is not an exact reproduction of the originally reported model. They are different rankings: the original report's test ROC-AUC is 0.8162 and the saved refit's is 0.8150. The original capacity counts are consistent with the original model's own top-10% table (recall 0.32517 = 863 / 2,654).

Against the committed seed refits (scikit-learn 1.9.0), the original counts sit inside the seed range at 5% and 10%, at its upper edge at 20%, and above every one of the 12 seeds at 30% (1,862 against a maximum of 1,861). One seed (seed 8) reaches the original test ROC-AUC of 0.8162 but captures 1,854 at 30%. The library version used for the original model was not recorded, so the cause of the 1,862 difference is not established.

All capture counts use the floor rule; only the rank-based priority-band counts in the saved queue files use rounded boundaries (for example 1,256 for the top 5% band).

Sources: [`prioritization_capacity_results.csv`](../results/ba_disruption_decision_support_v03__prioritization_capacity_results.csv) (original counts), [`model_family_top10_v03.csv`](../results/ba_disruption_modeling_v03__model_family_top10_v03.csv), [`model_family_comparison_v03.csv`](../results/ba_disruption_modeling_v03__model_family_comparison_v03.csv) (original test AUCs), [`event_cv_summary.csv`](../results/ba_disruption_model_validation_v03__event_cv_summary.csv) (CV), [`point_estimates_test.csv`](../followup_outputs/point_estimates_test.csv) (saved refit), [`hgb_seed_spread.csv`](../followup_outputs/hgb_seed_spread.csv) (seed refits), [`final_decision_support_readme_v03.md`](../results/ba_disruption_decision_support_v03__final_decision_support_readme_v03.md) (refit provenance).

## How the CV ± was established

The `±` on the original 5-fold CV figures is the sample standard deviation (n − 1 denominator) across the 5 folds, not a standard error. Recomputing it from the per-fold values in [`event_cv_fold_metrics.csv`](../results/ba_disruption_model_validation_v03__event_cv_fold_metrics.csv) matches `ROC_AUC_std` in `event_cv_summary.csv` to six decimal places. The original CV code was not recovered, so this definition is inferred from that match. The saved out-of-fold predictions ([`oof_predictions.csv`](../results/ba_disruption_model_validation_v03__oof_predictions.csv)) have 100,072 rows, one per itinerary, so the original v0.3 5-fold CV appears to have covered all events, including the 2,500 test events. The fold assignment was not stored (the file has no fold or event column), so this can't be confirmed. The model comparison's cross-validation, by contrast, used training events only.

## Selection-rule ordering

The README says the model comparison's decision rule was fixed before the full comparison run. The evidence is in the code: `model_selection_v03.py` writes the rule text into `model_selection_outputs/run_config.json` (`check_or_write_config`) before it fits the first fold (`run_comparison`), and on a resumed run it refuses to continue if the recorded configuration differs. The rule text recorded in `run_config.json` matches `WINNER_RULE` in the committed script. Git can't show this ordering, because the script and its outputs were committed together. A quick smoke run (`--quick`, written to the gitignored `model_selection_outputs_quick/`) existed before the full run.

## Export-environment versions

[`environment/relevant_versions.json`](../environment/relevant_versions.json) records the saved export environment: Python 3.13.5, NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0, with SciPy 1.17.0 and joblib 1.5.3. This differs from the versions pinned in `requirements.txt` and recorded by the model-comparison runs. `requirements.txt` does not recreate the historical export or the original v0.3 training environment.

## Per-event rounding rule

An earlier follow-up note ([`followup_v03_results__README.md`](../results/followup_v03_results__README.md), also quoted in `DOCUMENTED_CODE_FRAGMENTS.md`) documents a per-event rule for a logistic per-event ranking: floor(n × capacity) with at least one passenger per event. With 5–15 passengers per event, that rule always selects more than 5% of an event's passengers at the 5% level. The README's capture figures use the global rule instead, and figures from the two rules are not comparable.

## Same-day flag

The earlier follow-up note reports 97.044% agreement between the same-day flag and `best_recovery_arrival_minutes < 1440`. This was recomputed on the 39,378 missed itineraries that have a recovery value (97.6% over all 48,554 missed itineraries, counting itineraries with no recovery value as not < 1440). `sensitivity_alt_target_v03.py` finds that `original_final_arrival_minutes + best_recovery_arrival_minutes < 1440` matches the flag on all 39,378. The same note says the two fields are not on a directly verifiable common clock. No document in this repository defines the units of either column, so the two readings are unreconciled, and neither should be substituted for the frozen target definition. The headline results use the stored `severe_connection_disruption` flag and do not depend on this. The 180-minute alternative-target analysis assumes the second reading and would be invalid if it were wrong. Resolving this needs the original generator source or a written definition of both columns.

## Connection-miss rule

A rule inferred from the data, `event_actual_delay_minutes + actual_transfer_minutes > connection_buffer_minutes`, disagrees with the stored `original_connection_missed` label on 316 of 100,072 rows (0.3158%). The rule is not in the specification or source notes, and it uses outcome columns the models never see. All 316 have a stored label of 0 where the rule predicts 1; none has a recovery value, all have a severe label of 0, and 75 are in the test rows. These are disagreements between an inferred rule and the stored label, not confirmed labelling errors, and this check does not establish which interpretation is correct. The stored labels and the frozen target are unchanged, and the check does not show that the 75 test-row exceptions affect any model's predictions. `check_connection_miss_rule.py` reproduces it.

## Five-run shuffle row

[`ablations_hgb.csv`](../followup_outputs/ablations_hgb.csv) contains a smaller 5-run shuffled-label row (test ROC-AUC 0.455 ± 0.042, where the ± is a population standard deviation, ddof=0). It is not used: single runs vary because test passengers cluster in 2,500 events, so the README reports the 40-fit check in `shuffle_check_40.csv` instead.

## Logistic counts: 1,832 against 1,834

At 30% capacity, the earlier logistic follow-up ([`followup_v03_results__logistic_capacity_global.csv`](../results/followup_v03_results__logistic_capacity_global.csv)) captured 1,832 severe cases, while the balanced logistic refit in `point_estimates_test.csv` captures 1,834. The counts at 5%, 10% and 20% are the same in both (461, 799 and 1,356). The earlier refit's preprocessing is not documented, while `reproduce_followup_v03.py` adds a `StandardScaler`; the cause of the 2-case difference was not established.

## HGB row-level scores

The earlier follow-up note says HGB and simple-rule row-level scores were not available as saved artifacts, so it computed no HGB or simple-rule bootstrap or per-event ranking. The later follow-up (`reproduce_followup_v03.py`) uses the `risk_score` column of the saved HGB refit queue ([`priority_queue_evaluation_v03.csv`](../results/ba_disruption_decision_support_v03__priority_queue_evaluation_v03.csv)) and the reconstructed simple rule. Row-level predictions from the original HGB model remain unrecovered.

## MANIFEST scope

`MANIFEST.csv` and `MANIFEST.json` cover the original v0.3 export only and were not updated for later files: the follow-up and model-comparison scripts, their output folders, `audit_outputs/`, `docs/` and `tests/` are not listed. The README has been rewritten since the MANIFEST was generated, so its recorded checksum no longer matches.

## Booking-context columns and the Forage dataset

A check against my local copy of the Forage British Airways `customer_booking.csv` (not included in this repository) found that every `booking_origin` value in the passenger file occurs there, and that only 1,164 of 100,072 rows match a Forage row on all seven shared fields (`num_passengers`, `trip_type`, `purchase_lead`, `length_of_stay`, `flight_hour`, `flight_day`, `booking_origin`). This suggests the values were sampled field by field rather than copied as whole rows. `route_origin` and `route_dest` both hold six-letter values of the Forage `route` column (for example `AKLKUL`), not three-letter airport codes.
