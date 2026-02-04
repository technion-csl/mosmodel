#!/usr/bin/env python3
import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

def round_up(num, to_nearest_num):
    return math.ceil(num / to_nearest_num) * to_nearest_num

def plotModels(df: pd.DataFrame, models, output_pdf: Path):
    error_suffix = "_error"

    # Build max-errors table safely
    rows = []
    for m in models:
        col = m + error_suffix
        if col not in df.columns:
            print(f"[WARN] missing column {col}, using NaN", file=sys.stderr)
            max_error_pct = np.nan
        else:
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            max_error_pct = float(s.abs().max(skipna=True)) * 100 if not s.dropna().empty else 0.0

        rows.append({"model": m, "max-error": max_error_pct})

    max_errors_df = pd.DataFrame(rows)

    # Write CSV next to PDF
    csv_output = output_pdf.with_suffix(".csv")
    max_errors_df.to_csv(csv_output, index=False, float_format="%.3f")

    # Choose y-limit safely (avoid 0 / inf)
    maxv = pd.to_numeric(max_errors_df["max-error"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()
    if not np.isfinite(maxv) or maxv <= 0:
        bar_top = 10
    else:
        bar_top = max(10, round_up(maxv, 10))

    fig, ax = plt.subplots(figsize=(4, 3))
    ind = np.arange(len(models))
    ax.bar(ind, max_errors_df["max-error"])

    ax.set_ylabel("max absolute errors [%]")
    ax.set_xticks(ind)
    ax.set_xticklabels(models, rotation=-30, ha="left")
    ax.grid(axis="y")
    ax.set_ylim(0, bar_top)

    # IMPORTANT: bounded number of ticks (no huge np.arange)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))

    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)  # important when generating multiple figures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--errors", required=True,
                    help="CSV file containing <model>_error columns")
    ap.add_argument("-t", "--plot_title", default="Undefined", type=str,
                    help="(optional) plot title")
    # Accept BOTH names so your Make command works
    ap.add_argument("-o", "--output", "--output_dir", dest="output_dir", default="./",
                    help="output directory")
    args = ap.parse_args()

    df = pd.read_csv(args.errors)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plotModels(df, ["basu", "pham", "gandhi", "yaniv"], out_dir / "linear_models.pdf")
    plotModels(df, ["poly1", "poly2", "poly3", "mosmodel"], out_dir / "mosalloc_models.pdf")

if __name__ == "__main__":
    main()
