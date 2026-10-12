"""
List interface strings that have no translation yet.

    python tools/i18n_check.py tr          # missing Turkish entries
    python tools/i18n_check.py tr --write  # add them (English as placeholder)

Strings are collected from ``tr("...")`` calls under src/ui plus the dynamic
keys listed below (event kinds, affiliations, ...).
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOGS = ROOT / "src" / "ui" / "assets" / "i18n"

DYNAMIC_KEYS = [
    "LIVE", "PAUSED", "REPLAY", "EDIT",
    "first detection", "track initiated", "track confirmed", "track lost", "masked",
    "unmasked", "jammed", "burn through", "deployed",
    "hostile", "friend", "neutral", "unknown",
    "aircraft", "fighter", "bomber", "missile", "ship", "drone", "helicopter", "chaff", "decoy",
    "Rural", "Urban", "Forest", "Desert", "Mountain", "Sea",
    "Hostile", "Friend", "Neutral", "Unknown",
    "PPI", "B-Scope", "RHI", "A-Scope", "Range–Doppler", "3D",
]


def interface_strings():
    found = set(DYNAMIC_KEYS)
    for path in (ROOT / "src" / "ui").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "tr"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.add(node.args[0].value)
    return sorted(found)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    code = sys.argv[1] if len(sys.argv) > 1 else "tr"
    path = CATALOGS / f"{code}.json"
    catalog = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [s for s in interface_strings() if s not in catalog]
    for text in missing:
        print(text)
    if "--write" in sys.argv and missing:
        catalog.update({text: text for text in missing})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(catalog.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(missing)} missing of {len(interface_strings())}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
