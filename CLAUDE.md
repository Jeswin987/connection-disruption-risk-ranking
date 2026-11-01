# CLAUDE.md

Working rules for Claude Code sessions in this repository. README.md is the source of truth for results; docs/PROVENANCE.md explains what was and wasn't recovered from the original v0.3 work.

## Setup and tests
- Python 3.13. Create a venv, then `python -m pip install -r requirements-dev.txt`.
- Run the full suite with `python -m pytest -q` (one HGB refit; no other model fitting).
- Shell: Git Bash on Windows. Check that the prompt shows `$`, not PowerShell.

## Never do these
- Never modify, regenerate or delete committed data or results: data/, event_split/, predictions/, results/, followup_outputs/, model_selection_outputs/, final_check_outputs/, audit_outputs/, source_status/, environment/, MANIFEST.csv, MANIFEST.json.
- Never use the test events (event_split/test_event_ids.csv) to choose or tune a model. They were already inspected during development; see the README.
- Never run the full model comparison (model_selection_v03.py or final_check_v03.py without --quick; about 5 hours for the two together) unless asked.
- Never add a licence file; the repository deliberately has none.
- Never change the line-ending rules in .gitattributes; the MANIFEST checksum tests depend on LF checkouts.

## Running analyses
- The three follow-up analysis scripts (reproduce_followup_v03.py, sensitivity_alt_target_v03.py, shuffle_check_v03.py) need an explicit out_dir that is not followup_outputs/, e.g. `python reproduce_followup_v03.py . rerun_outputs`. check_connection_miss_rule.py takes only the package folder. Top-level folders named rerun_* are gitignored.
- Compare reruns with `git diff --no-index --ignore-cr-at-eol --exit-code followup_outputs rerun_outputs`.
- Anything that refits HGB can vary with library version; keep the requirements.txt pins unless changing them is the task.

## Workflow
- Work on a branch, never directly on master. Never push unless asked.
- One scoped commit per change. Stage files by explicit name only; never `git add -A`, `git add .`, `git add -u` or `git commit -a`.
- Run the full suite before every commit, and don't commit if anything fails.
- Stop after each commit and wait for "go".
- For real bugs, write the failing test first and show it failing.
- Every number in README.md must trace to a committed file. If a README number changes, update the test in tests/test_headline_numbers.py that guards it. The README link test must keep passing.
- Merge with `git merge --ff-only`, then run a gitleaks scan before pushing.
