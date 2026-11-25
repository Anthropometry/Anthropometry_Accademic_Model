"""Copy the measurement catalogue from the AnthropometryDC app into this repository (S7-0).

Usage:
    python scripts/sync_catalog.py APP/src/lib/measurements.ts --source "repo branch@commit"

Why a copy and not a live link: the model must be reproducible against a fixed list, so the copy is
stamped with where it came from, and changing it is a reviewed commit. The app's file is the source of
truth; this script only reads it (code, never data).
"""

import argparse
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "anthro" / "data" / "measurements.json"


def parse(ts: str) -> dict:
    pattern = r"\{ code: '([a-z_]+)', label: '([^']+)', how: '([^']+)', adult: \[(\d+), (\d+)\] \}"
    entries = re.findall(pattern, ts)
    if len(entries) != 25:
        raise SystemExit(f"expected 25 measurements in the app's CATALOGUE, found {len(entries)}")

    def code_list(name: str) -> list[str]:
        m = re.search(rf"const {name}: readonly Code\[\] = \[([^\]]*)\]", ts)
        if not m:
            raise SystemExit(f"{name} not found in measurements.ts")
        return re.findall(r"'([a-z_]+)'", m.group(1))

    def code_set(name: str) -> list[str]:
        m = re.search(rf"export const {name}: ReadonlySet<Code> = new Set\(\[([^\]]*)\]\)", ts)
        if not m:
            raise SystemExit(f"{name} not found in measurements.ts")
        return re.findall(r"'([a-z_]+)'", m.group(1))

    male_only, female_only = set(code_list("MALE_ONLY")), set(code_list("FEMALE_ONLY"))
    personal, double = set(code_set("INTIMATE")), set(code_set("DOUBLE"))
    measurements = []
    for code, label, how, lo, hi in entries:
        sexes = ["male"] if code in male_only else ["female"] if code in female_only else ["female", "male"]
        measurements.append({
            "code": code, "label": label, "how": how,
            "type": "girth" if code.endswith("_girth") else "length",
            "adult_range_cm": [int(lo), int(hi)],
            "sexes": sexes, "personal": code in personal, "taken_twice": code in double,
        })
    return {"measurements": measurements}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ts", help="path to AnthropometryDC src/lib/measurements.ts")
    ap.add_argument("--source", required=True,
                    help='e.g. "Anthropometry/AnthropometryDC f8-admin-export@807cd85"')
    a = ap.parse_args()
    cat = parse(Path(a.ts).read_text())
    cat = {"catalogue_version": f"anthropometrydc-measurements@{a.source.split('@')[-1]}",
           "source": a.source, "adults_only": True, **cat}
    OUT.write_text(json.dumps(cat, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(cat['measurements'])} measurements)")


if __name__ == "__main__":
    main()
