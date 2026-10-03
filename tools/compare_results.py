"""Compare the ``metrics`` sections of two result directories (reproducibility check).

Usage: python tools/compare_results.py DIR_A DIR_B [--rtol 0]

Timing, timestamps and environment fields are ignored. Exits non-zero on any difference.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def walk(a, b, path, rtol, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: missing on one side")
            else:
                walk(a[k], b[k], f"{path}/{k}", rtol, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", rtol, diffs)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= rtol * max(abs(a), abs(b))):
            diffs.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")


def main() -> int:
    da, db = Path(sys.argv[1]), Path(sys.argv[2])
    rtol = float(sys.argv[sys.argv.index("--rtol") + 1]) if "--rtol" in sys.argv else 0.0
    n_files, diffs = 0, []
    for fa in sorted(da.glob("*.json")):
        fb = db / fa.name
        if not fb.exists():
            diffs.append(f"{fa.name}: missing in {db}")
            continue
        ja, jb = json.loads(fa.read_text()), json.loads(fb.read_text())
        walk(ja["metrics"], jb["metrics"], fa.stem, rtol, diffs)
        walk(ja["config"], jb["config"], fa.stem + ":config", rtol, diffs)
        n_files += 1
    for d in diffs:
        print(d)
    print(f"{n_files} result files compared, {len(diffs)} differences (rtol = {rtol})")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
