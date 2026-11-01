"""Every relative Markdown link in README.md points to a path tracked by git."""
import re
import subprocess

import pytest

from conftest import ROOT

# Inline links and images: [text](target) or [text](target "title").
LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")


def tracked_paths() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8")
    files = {p for p in out.split("\0") if p}
    # Directories that contain tracked files are valid link targets too.
    dirs = {"/".join(p.split("/")[:i]) for p in files for i in range(1, p.count("/") + 1)}
    return files | dirs


def relative_links() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    links = []
    for target in LINK_RE.findall(text):
        if re.match(r"https?://", target, re.IGNORECASE) or target.startswith("#"):
            continue
        links.append(target)
    return links


LINKS = relative_links()
TRACKED = tracked_paths()


def test_readme_has_relative_links():
    assert LINKS


@pytest.mark.parametrize("target", LINKS)
def test_link_points_to_tracked_path(target):
    path = target.split("#", 1)[0].removeprefix("./").rstrip("/")
    # Exact-case string comparison, so a case mismatch fails even on Windows.
    assert path in TRACKED, f"README link target not tracked by git: {target}"
