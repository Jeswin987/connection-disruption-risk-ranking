"""The follow-up scripts refuse to run without a safe, explicit out_dir.

Each script runs with cwd and package_dir set to an empty tmp_path, so even
without the guard nothing can touch the real repository.
"""
import subprocess
import sys

import pytest

from conftest import ROOT

SCRIPTS = ["reproduce_followup_v03.py", "sensitivity_alt_target_v03.py", "shuffle_check_v03.py"]
MISSING_MSG = "out_dir is required"
PROTECTED_MSG = "refusing to write to the committed followup_outputs/ folder"


def run_script(script, args, tmp_path):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_missing_out_dir_is_rejected(script, tmp_path):
    result = run_script(script, [str(tmp_path)], tmp_path)
    assert result.returncode != 0
    assert MISSING_MSG in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("script", SCRIPTS)
def test_committed_followup_outputs_is_rejected(script, tmp_path):
    result = run_script(script, [str(tmp_path), "followup_outputs"], tmp_path)
    assert result.returncode != 0
    assert PROTECTED_MSG in result.stderr
    assert list(tmp_path.iterdir()) == []
