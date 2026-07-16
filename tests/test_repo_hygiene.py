"""Every source/config file that exists on disk must be git-tracked.

Guards against overly-broad .gitignore patterns: an unanchored `data/` once
ignored src/querent/data/ entirely — local tests passed (files on disk) while
every clone crashed on import. This test fails the moment a tracked tree and
the working tree diverge on .py/.yaml files.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tracked() -> set[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    return set(out.splitlines())


def test_all_source_files_tracked():
    tracked = _tracked()
    missing = []
    for pattern, root in (("*.py", "src"), ("*.py", "scripts"), ("*.py", "tests"),
                          ("*.yaml", "configs"), ("*", "docker")):
        for p in (REPO / root).rglob(pattern):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = str(p.relative_to(REPO))
                if rel not in tracked:
                    missing.append(rel)
    assert not missing, f"files on disk but NOT git-tracked (gitignore bite?): {missing}"
