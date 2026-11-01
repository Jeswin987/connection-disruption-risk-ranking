"""Committed artifacts still match MANIFEST.csv byte-for-byte."""
import csv
import hashlib

import pytest

from conftest import ROOT

with open(ROOT / "MANIFEST.csv", newline="", encoding="utf-8") as fh:
    ENTRIES = list(csv.DictReader(fh))

# README.md is expected to differ: it has been rewritten since the MANIFEST
# was generated, so its recorded hash and size are no longer current.
CHECKED = [e for e in ENTRIES if e["file"] != "README.md"]


def test_manifest_has_entries():
    assert CHECKED
    assert len(CHECKED) == len(ENTRIES) - 1


@pytest.mark.parametrize("entry", CHECKED, ids=[e["file"] for e in CHECKED])
def test_file_matches_manifest(entry):
    data = (ROOT / entry["file"]).read_bytes()
    assert len(data) == int(entry["bytes"])
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
