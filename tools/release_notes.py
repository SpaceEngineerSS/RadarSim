"""
Print the CHANGELOG section for one version (used by the release workflow).

    python tools/release_notes.py 4.0.0
"""

import re
import sys
from pathlib import Path


def main() -> int:
    version = sys.argv[1]
    text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(rf"^## \[?{re.escape(version)}\]?.*?$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not match:
        print(f"No changelog entry for {version}", file=sys.stderr)
        return 1
    sys.stdout.reconfigure(encoding="utf-8")
    print(match.group(1).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
