"""Regenerate every result file and figure, in dependency order.

Usage (from the repository root or from experiments/):
    python experiments/run_all.py            # recompute results, then plot
    python experiments/run_all.py --plot-only
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ORDER = [
    "ula_verification",
    "jammer_suppression",
    "snapshot_sensitivity",
    "loading_sensitivity",
    "steering_mismatch",
    "jammer_sweep",
    "range_doppler_demo",
    "cfar_false_alarm_check",
    "cfar_pd_snr",
    "integrated_array_radar",  # reads cfar_false_alarm_check results
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    t_all = time.perf_counter()
    for name in ORDER:
        mod = importlib.import_module(name)
        t0 = time.perf_counter()
        if not args.plot_only:
            mod.run()
        out = mod.plot()
        print(f"{name:28s} {time.perf_counter() - t0:7.1f} s  -> {out}")
    print(f"total {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
