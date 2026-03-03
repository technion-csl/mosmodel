#! /usr/bin/env python3

import argparse
from typing import Optional

import numpy as np
import pandas as pd


from performance_statistics import PerformanceStatistics


DERIVED = {"cpi", "stlb_mpki", "mpki"}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _load_csv_with_layout(path: str) -> pd.DataFrame:
    """Load CSV and ensure it has a 'layout' column.

    Common fallbacks:
      - If first column is 'Unnamed: 0' / 'index', treat it as 'layout'
      - If column is 'Layout', rename to 'layout'
    """
    df = pd.read_csv(path)

    if "layout" not in df.columns:
        if "Layout" in df.columns:
            df = df.rename(columns={"Layout": "layout"})
        else:
            c0 = df.columns[0] if len(df.columns) else None
            if c0 in ("index",) or (isinstance(c0, str) and c0.startswith("Unnamed")):
                df = df.rename(columns={c0: "layout"})

    if "layout" not in df.columns:
        raise ValueError(f"Input '{path}' must contain a 'layout' column. Found columns: {list(df.columns)}")

    return df


def _enrich_with_ps_getters(df: pd.DataFrame) -> pd.DataFrame:
    """Attach/overwrite key columns using PerformanceStatistics getters.

    This mirrors the approach in buildLinearModelsCoeffs.py:
      ps = PerformanceStatistics(df.copy()); then overwrite columns via getters.
    """
    df = df.copy()

    try:
        ps = PerformanceStatistics(df.copy())
    except TypeError:
        # Some older PS versions may not accept a DataFrame ctor.
        # In that case, keep df as-is; derived metrics may not be computable.
        return df

    if hasattr(ps, "getRuntime"):
        df["cpu-cycles"] = ps.getRuntime()

    if hasattr(ps, "getStlbMisses"):
        df["stlb_misses"] = ps.getStlbMisses()
        # Historical alias sometimes used elsewhere
        df["tlb_misses"] = df["stlb_misses"]

    if "instructions" not in df.columns and hasattr(ps, "getInstructions"):
        df["instructions"] = ps.getInstructions()

    return df


def _metric_from_df(df: pd.DataFrame, metric: str) -> pd.Series:
    """Compute/lookup a metric from an enriched df (row-wise)."""
    if metric in ("mpki", "stlb_mpki", "cpi"):
        # Only require what is actually needed for the specific derived metric.
        if "instructions" not in df.columns:
            raise ValueError(f"Cannot compute '{metric}': missing 'instructions' column.")
        instr = _to_num(df["instructions"])
        ok = instr.notna() & (instr > 0)

        out = pd.Series(np.nan, index=df.index, dtype=float)

        if metric == "cpi":
            if "cpu-cycles" not in df.columns:
                raise ValueError("Cannot compute 'cpi': missing 'cpu-cycles' column.")
            cycles = _to_num(df["cpu-cycles"])
            out.loc[ok] = cycles.loc[ok] / instr.loc[ok]
            return out

        # MPKI variants
        if "stlb_misses" not in df.columns:
            raise ValueError(f"Cannot compute '{metric}': missing 'stlb_misses' column.")
        misses = _to_num(df["stlb_misses"])
        out.loc[ok] = (misses.loc[ok] * 1000.0) / instr.loc[ok]
        return out

    if metric in df.columns:
        return _to_num(df[metric])

    # Allow alias
    if metric == "tlb_misses" and "stlb_misses" in df.columns:
        return _to_num(df["stlb_misses"])

    raise ValueError(f"Unknown metric '{metric}' (not derived and not a CSV column).")


def _std_for_basic_metric(std_df: pd.DataFrame, metric: str) -> pd.Series:
    """Read std for a *basic* metric from std_df (no approximations)."""
    if metric in DERIVED:
        raise ValueError("Internal error: _std_for_basic_metric called for derived metric.")

    if metric in std_df.columns:
        return _to_num(std_df[metric])

    if metric == "tlb_misses" and "stlb_misses" in std_df.columns:
        return _to_num(std_df["stlb_misses"])

    raise ValueError(f"std CSV is missing column '{metric}'. Found columns: {list(std_df.columns)}")


def _sample_std_by_group(v: pd.Series, group: pd.Series) -> pd.Series:
    """Sample std (ddof=1) per group; return 0.0 for groups of size 1."""
    tmp = pd.DataFrame({"g": group, "v": v})
    tmp = tmp[tmp["v"].notna()]
    std = tmp.groupby("g")["v"].std(ddof=1)
    cnt = tmp.groupby("g")["v"].count()
    std = std.reindex(cnt.index)
    std.loc[cnt <= 1] = 0.0
    return std


def read_single(mean_file: str, std_file: Optional[str], y_metric: str, x_metric: str) -> pd.DataFrame:
    """Read one mean/repeats CSV (+ optional std CSV) and return a plot-ready DataFrame."""

    # ---- Read mean/repeats file (single file) ----
    mean_df = _enrich_with_ps_getters(_load_csv_with_layout(mean_file))
    mean_df["layout"] = mean_df["layout"].astype(str)

    repeats_mode = std_file is not None  # per user's pipeline: std_file => mean_file contains repeats

    if not repeats_mode:
        # Mean-mode: one row per layout
        x_vals = _metric_from_df(mean_df, x_metric)
        y_vals = _metric_from_df(mean_df, y_metric)

        out = pd.DataFrame({"layout": mean_df["layout"], x_metric: x_vals, y_metric: y_vals})

        # Keep deterministic order
        out = out.sort_values(["layout"]).reset_index(drop=True)
        return out

    # ---- Repeats-mode ----
    # Compute x,y per repeat row then aggregate per layout
    x_rep = _metric_from_df(mean_df, x_metric)
    y_rep = _metric_from_df(mean_df, y_metric)

    rep = pd.DataFrame({"layout": mean_df["layout"], "x": x_rep, "y": y_rep})

    mean_agg = (
        rep.groupby("layout", as_index=False)
        .mean(numeric_only=True)
        .rename(columns={"x": x_metric, "y": y_metric})
    )

    # ---- Std handling ----
    # - for derived metrics: compute sample std directly from repeats
    # - for basic metrics: take std from std_file
    std_cols: dict[str, pd.Series] = {}

    if x_metric in DERIVED:
        std_cols[f"{x_metric}_std"] = _sample_std_by_group(rep["x"], rep["layout"])
    if y_metric in DERIVED:
        std_cols[f"{y_metric}_std"] = _sample_std_by_group(rep["y"], rep["layout"])

    std_df = _enrich_with_ps_getters(_load_csv_with_layout(std_file))
    std_df["layout"] = std_df["layout"].astype(str)

    # If std_df accidentally has repeats, keep the first row per layout (std should be 1 row/layout).
    std_df = std_df.groupby("layout", as_index=False).first()
    std_df = std_df.set_index("layout", drop=True)

    if x_metric not in DERIVED:
        std_cols[f"{x_metric}_std"] = _std_for_basic_metric(std_df, x_metric)
    if y_metric not in DERIVED:
        std_cols[f"{y_metric}_std"] = _std_for_basic_metric(std_df, y_metric)

    # Merge means + stds on layout
    out = mean_agg.set_index("layout", drop=True)

    for col, series in std_cols.items():
        out[col] = series.reindex(out.index)

    out = out.reset_index()  # keep layout column
    # Order: layout first, then x/y, then stds if present
    cols = ["layout", x_metric, y_metric]
    for c in (f"{x_metric}_std", f"{y_metric}_std"):
        if c in out.columns:
            cols.append(c)
    out = out[cols]

    # Sort by x metric for plotting, but keep layout column
    out = out.sort_values(x_metric).reset_index(drop=True)
    return out


def _normalize_inplace(df: pd.DataFrame, x_metric: str, y_metric: str, mode: Optional[str]) -> pd.DataFrame:
    if not mode:
        return df

    df = df.copy()

    max_y = pd.to_numeric(df[y_metric], errors="coerce").max()
    max_x = pd.to_numeric(df[x_metric], errors="coerce").max()

    ystd = f"{y_metric}_std"
    xstd = f"{x_metric}_std"

    if mode == "by-y":
        if max_y and np.isfinite(max_y) and max_y != 0:
            df[y_metric] = df[y_metric] / max_y
            df[x_metric] = df[x_metric] / max_y
            if ystd in df.columns:
                df[ystd] = df[ystd] / max_y
            if xstd in df.columns:
                df[xstd] = df[xstd] / max_y
        return df

    if mode == "separate":
        if max_y and np.isfinite(max_y) and max_y != 0:
            df[y_metric] = df[y_metric] / max_y
            if ystd in df.columns:
                df[ystd] = df[ystd] / max_y
        if max_x and np.isfinite(max_x) and max_x != 0:
            df[x_metric] = df[x_metric] / max_x
            if xstd in df.columns:
                df[xstd] = df[xstd] / max_x
        return df

    raise ValueError(f"Unknown normalize mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-m",
        "--mean_file",
        default="mean.csv",
        help=(
            "input CSV file. If --std_file is provided, it is interpreted as a per-repeat file "
            "(multiple rows per layout)."
        ),
    )
    ap.add_argument(
        "-s",
        "--std_file",
        default=None,
        help="optional CSV file containing stdev values for basic counters (one row per layout).",
    )
    ap.add_argument("-o", "--output", required=True, help="output CSV file")

    ap.add_argument(
        "-n",
        "--normalize",
        choices=["by-y", "separate"],
        default=None,
        help="how to normalize the data columns",
    )

    ap.add_argument(
        "-x",
        "--x-metric",
        choices=["walk_cycles", "tlb_misses", "stlb_mpki", "mpki", "cpi"],
        default="tlb_misses",
        help="metric to use for x-axis (supports derived: mpki/stlb_mpki, cpi)",
    )
    ap.add_argument(
        "-y",
        "--y-metric",
        default="cpu-cycles",
        help="metric to use for y-axis (raw column name OR derived: mpki/stlb_mpki, cpi)",
    )

    args = ap.parse_args()

    output_df = read_single(args.mean_file, args.std_file, args.y_metric, args.x_metric)
    output_df = _normalize_inplace(output_df, args.x_metric, args.y_metric, args.normalize)

    output_df.to_csv(args.output, float_format="%.3f", index=False)


if __name__ == "__main__":
    main()
