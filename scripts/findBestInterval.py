
#! /usr/bin/env python3
"""
Find the earliest / shortest instruction interval whose (CPI, MPKI) best matches
the whole-run point from perf stat interval output.

Supports:
  - a single perf.out file
  - a directory containing repeat*/perf.out subdirectories

Search strategy:
  1) Build 1-minute bins (default resolution = 60s) from 1-second perf intervals.
  2) Try all candidate window lengths in minutes.
  3) For each candidate window, compare its CPI/MPKI against the whole run.
  4) Mark a window "acceptable" if mean relative CPI/MPKI errors across repeats
     are under thresholds.
  5) Select by:
       acceptable first,
       then earliest start,
       then shortest duration,
       then lowest mean combined error.

Returns:
  - the chosen time window
  - per-repeat instruction bounds I_start / I_end
  - per-repeat and aggregate CPI / MPKI / errors

MPKI is computed as:
  1000 * (dtlb_load_misses.walk_completed + dtlb_store_misses.walk_completed) / instructions
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_CANDIDATE_INTERVALS_MIN = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]

CPU_CYCLES_COL = "cpu-cycles"
INSTRUCTIONS_COL = "instructions"
LOAD_WALK_COMPLETED_COL = "dtlb_load_misses.walk_completed"
STORE_WALK_COMPLETED_COL = "dtlb_store_misses.walk_completed"


def readPerfIntervalFile(perf_out_path, skiprows=2):
    """Parse perf stat output produced with --interval-print=1000 (CSV).

    Expected columns (by position):
      0: time
      1: counter_value
      3: counter_name

    Returns:
        wide_df: DataFrame indexed by time, with one column per counter_name.
                Values are per-interval counts (not cumulative).
    """
    try:
        df = pd.read_csv(
            perf_out_path,
            skiprows=skiprows,
            usecols=[0, 1, 3],
            names=["time", "counter_value", "counter_name"],
            na_values="<not counted>",
        )
    except IOError:
        return None
    except Exception as e:
        raise ValueError(f"Could not read perf interval CSV: {perf_out_path} ({e})")

    # Clean / normalize
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["counter_value"] = pd.to_numeric(df["counter_value"], errors="coerce")
    df["counter_name"] = df["counter_name"].astype(str).str.strip()

    df = df.dropna(subset=["time", "counter_name"]).sort_values("time")

    # Pivot to wide
    wide = df.pivot(index="time", columns="counter_name", values="counter_value")

    # Strip perf suffixes like ':u'
    wide.columns = [str(c).replace(":u", "") for c in wide.columns]

    # Ensure numeric dtype
    wide = wide.apply(pd.to_numeric, errors="coerce")

    return wide


def find_perf_files(path: Path, perf_filename: str = "perf.out") -> List[Path]:
    """Return one or more perf.out paths from either a file or a repeats directory."""
    path = path.resolve()

    if path.is_file():
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    direct_perf = path / perf_filename
    if direct_perf.is_file():
        return [direct_perf]

    perf_files = sorted(p for p in path.glob(f"repeat*/{perf_filename}") if p.is_file())
    if perf_files:
        return perf_files

    raise FileNotFoundError(
        f"Could not find {perf_filename} directly under {path} "
        f"or under repeat*/{perf_filename}"
    )


def validate_columns(df: pd.DataFrame, perf_path: Path) -> None:
    required = [
        CPU_CYCLES_COL,
        INSTRUCTIONS_COL,
        LOAD_WALK_COMPLETED_COL,
        STORE_WALK_COMPLETED_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required counters in {perf_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def build_resolution_df(
    wide_df: pd.DataFrame,
    resolution_sec: int = 60,
) -> pd.DataFrame:
    """
    Convert 1-second-ish interval rows into coarser bins using row-order grouping.
    This is more robust than relying on slightly noisy timestamps like 60.0003.
    """
    if wide_df is None or wide_df.empty:
        raise ValueError("Empty perf interval dataframe.")

    work = wide_df.copy().sort_index()
    work = work.reset_index().rename(columns={"index": "time"})
    work["sample_idx"] = np.arange(len(work), dtype=np.int64)

    samples_per_bin = max(1, int(resolution_sec))
    work["bin_idx"] = work["sample_idx"] // samples_per_bin

    grouped = work.groupby("bin_idx", sort=True)

    numeric_cols = [c for c in work.columns if c not in {"time", "sample_idx", "bin_idx"}]
    binned = grouped[numeric_cols].sum(min_count=1)

    # Helpful metadata for the chosen bins.
    binned["time_start_sec"] = grouped["time"].min()
    binned["time_end_sec"] = grouped["time"].max()
    binned["samples_in_bin"] = grouped.size()

    # If some counter is absent in a whole bin, sum(min_count=1) returns NaN.
    # Keep mandatory counters strict, fill optional missing counters with 0.
    for col in binned.columns:
        if col in {"time_start_sec", "time_end_sec", "samples_in_bin"}:
            continue
        binned[col] = pd.to_numeric(binned[col], errors="coerce")

    # Require the main counters to exist for every bin we plan to use.
    mandatory = [CPU_CYCLES_COL, INSTRUCTIONS_COL, LOAD_WALK_COMPLETED_COL, STORE_WALK_COMPLETED_COL]
    if binned[mandatory].isna().any().any():
        bad_bins = binned.index[binned[mandatory].isna().any(axis=1)].tolist()
        raise ValueError(
            f"Mandatory counters became NaN after binning in bins {bad_bins}. "
            f"This usually means missing perf samples/counters."
        )

    return binned


def cpi_from_counts(cycles: float, instructions: float) -> float:
    if not np.isfinite(cycles) or not np.isfinite(instructions) or instructions <= 0:
        return math.nan
    return float(cycles) / float(instructions)


def mpki_from_counts(load_walk_completed: float, store_walk_completed: float, instructions: float) -> float:
    misses = float(load_walk_completed) + float(store_walk_completed)
    if not np.isfinite(misses) or not np.isfinite(instructions) or instructions <= 0:
        return math.nan
    return 1000.0 * misses / float(instructions)


@dataclass
class RepeatData:
    name: str
    perf_path: Path
    second_df: pd.DataFrame
    bin_df: pd.DataFrame
    full_cycles: float
    full_instructions: float
    full_load_walk_completed: float
    full_store_walk_completed: float
    full_cpi: float
    full_mpki: float
    bin_cum_instructions: np.ndarray
    resolution_sec: int

    @property
    def n_bins(self) -> int:
        return len(self.bin_df)


def make_repeat_data(perf_path: Path, resolution_sec: int, skiprows: int = 2) -> RepeatData:
    second_df = readPerfIntervalFile(perf_path, skiprows=skiprows)
    if second_df is None or second_df.empty:
        raise ValueError(f"Could not read perf data from {perf_path}")

    validate_columns(second_df, perf_path)
    bin_df = build_resolution_df(second_df, resolution_sec=resolution_sec)

    full_cycles = float(second_df[CPU_CYCLES_COL].sum())
    full_instructions = float(second_df[INSTRUCTIONS_COL].sum())
    full_load = float(second_df[LOAD_WALK_COMPLETED_COL].sum())
    full_store = float(second_df[STORE_WALK_COMPLETED_COL].sum())
    full_cpi = cpi_from_counts(full_cycles, full_instructions)
    full_mpki = mpki_from_counts(full_load, full_store, full_instructions)

    bin_instructions = pd.to_numeric(bin_df[INSTRUCTIONS_COL], errors="coerce").to_numpy(dtype=float)
    bin_cum_instructions = np.concatenate(([0.0], np.cumsum(bin_instructions)))

    name = perf_path.parent.name if perf_path.parent.name.startswith("repeat") else perf_path.stem

    return RepeatData(
        name=name,
        perf_path=perf_path,
        second_df=second_df,
        bin_df=bin_df,
        full_cycles=full_cycles,
        full_instructions=full_instructions,
        full_load_walk_completed=full_load,
        full_store_walk_completed=full_store,
        full_cpi=full_cpi,
        full_mpki=full_mpki,
        bin_cum_instructions=bin_cum_instructions,
        resolution_sec=resolution_sec,
    )


def rel_err(value: float, ref: float) -> float:
    if not np.isfinite(value) or not np.isfinite(ref) or ref <= 0:
        return math.inf
    return abs(value - ref) / ref


def combined_error(cpi_err: float, mpki_err: float) -> float:
    return cpi_err + mpki_err


def summarize_window_for_repeat(r: RepeatData, start_bin: int, length_bins: int) -> Dict[str, float]:
    end_bin = start_bin + length_bins
    window = r.bin_df.iloc[start_bin:end_bin]

    cycles = float(window[CPU_CYCLES_COL].sum())
    instructions = float(window[INSTRUCTIONS_COL].sum())
    load_wc = float(window[LOAD_WALK_COMPLETED_COL].sum())
    store_wc = float(window[STORE_WALK_COMPLETED_COL].sum())

    cpi = cpi_from_counts(cycles, instructions)
    mpki = mpki_from_counts(load_wc, store_wc, instructions)

    cpi_err = rel_err(cpi, r.full_cpi)
    mpki_err = rel_err(mpki, r.full_mpki)
    score = combined_error(cpi_err, mpki_err)

    start_time_sec = float(window["time_start_sec"].iloc[0])
    end_time_sec = float(window["time_end_sec"].iloc[-1])

    I_start = float(r.bin_cum_instructions[start_bin])
    I_end = float(r.bin_cum_instructions[end_bin])

    return {
        "start_bin": start_bin,
        "end_bin": end_bin,
        "start_time_sec": start_time_sec,
        "end_time_sec": end_time_sec,
        "duration_bins": length_bins,
        "duration_sec": length_bins * float(window["samples_in_bin"].iloc[0]) if "samples_in_bin" in window.columns else math.nan,
        "window_cycles": cycles,
        "window_instructions": instructions,
        "window_load_walk_completed": load_wc,
        "window_store_walk_completed": store_wc,
        "window_cpi": cpi,
        "window_mpki": mpki,
        "full_cpi": r.full_cpi,
        "full_mpki": r.full_mpki,
        "cpi_rel_err": cpi_err,
        "mpki_rel_err": mpki_err,
        "score": score,
        "I_start": I_start,
        "I_end": I_end,
    }


def aggregate_candidate(
    repeats: Sequence[RepeatData],
    start_bin: int,
    length_bins: int,
    duration_min_requested: int,
    cpi_tol: float,
    mpki_tol: float,
) -> Dict:
    per_repeat = {}
    cpi_errs = []
    mpki_errs = []
    scores = []

    for r in repeats:
        s = summarize_window_for_repeat(r, start_bin, length_bins)
        per_repeat[r.name] = s
        cpi_errs.append(s["cpi_rel_err"])
        mpki_errs.append(s["mpki_rel_err"])
        scores.append(s["score"])

    mean_cpi_err = float(np.mean(cpi_errs))
    mean_mpki_err = float(np.mean(mpki_errs))
    mean_score = float(np.mean(scores))

    resolution_sec = repeats[0].resolution_sec
    start_sec = start_bin * resolution_sec
    end_sec = (start_bin + length_bins) * resolution_sec

    candidate = {
        "start_bin": start_bin,
        "end_bin": start_bin + length_bins,
        "length_bins": length_bins,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "start_min": start_sec / 60.0,
        "end_min": end_sec / 60.0,
        "duration_sec": length_bins * resolution_sec,
        "duration_min": (length_bins * resolution_sec) / 60.0,
        "duration_min_requested": duration_min_requested,
        "mean_cpi_rel_err": mean_cpi_err,
        "mean_mpki_rel_err": mean_mpki_err,
        "mean_score": mean_score,
        "max_cpi_rel_err": float(np.max(cpi_errs)),
        "max_mpki_rel_err": float(np.max(mpki_errs)),
        "median_score": float(np.median(scores)),
        "acceptable": (mean_cpi_err <= cpi_tol) and (mean_mpki_err <= mpki_tol),
        "per_repeat": per_repeat,
    }
    return candidate


def candidate_sort_key(c: Dict) -> Tuple:
    # acceptable first, then earliest finishing time, then shortest, then smallest error
    return (
        0 if c["acceptable"] else 1,
        c["end_sec"],
        c["duration_sec"],
        c["mean_score"],
        c["mean_cpi_rel_err"],
        c["mean_mpki_rel_err"],
    )


def scan_candidates(
    repeats: Sequence[RepeatData],
    candidate_intervals_min: Sequence[int],
    cpi_tol: float,
    mpki_tol: float,
    require_all_repeats: bool = True,
) -> Tuple[Dict, List[Dict]]:
    if not repeats:
        raise ValueError("No repeats to scan.")

    min_bins = min(r.n_bins for r in repeats)
    if min_bins <= 0:
        raise ValueError("No usable bins after resolution conversion.")

    all_candidates: List[Dict] = []
    resolution_sec = repeats[0].resolution_sec

    for duration_min in candidate_intervals_min:
        L = int(round((float(duration_min) * 60.0) / float(resolution_sec)))
        if L <= 0 or L > min_bins:
            continue

        for start_bin in range(0, min_bins - L + 1):
            candidate = aggregate_candidate(
                repeats=repeats,
                start_bin=start_bin,
                length_bins=L,
                duration_min_requested=int(duration_min),
                cpi_tol=cpi_tol,
                mpki_tol=mpki_tol,
            )
            all_candidates.append(candidate)

    if not all_candidates:
        min_available_sec = min_bins * repeats[0].resolution_sec
        raise ValueError(
            "No valid candidate windows. "
            f"Min available duration across repeats is only {min_bins} bins "
            f"(~{min_available_sec/60.0:.2f} minutes)."
        )

    all_candidates.sort(key=candidate_sort_key)
    best = all_candidates[0]
    return best, all_candidates


def err_pct(value: float) -> float:
    if not np.isfinite(value):
        return math.inf
    return 100.0 * float(value)


def rounded(value: float, digits: int) -> float:
    if not np.isfinite(value):
        return value
    return round(float(value), digits)


def rounded_pct(value: float) -> int | float:
    if not np.isfinite(value):
        return value
    return int(round(err_pct(value)))


def _per_repeat_report(rep: Dict) -> Dict:
    return {
        "I_start": int(round(rep["I_start"])),
        "I_end": int(round(rep["I_end"])),
        "start_time_sec": rounded(rep["start_time_sec"], 3),
        "end_time_sec": rounded(rep["end_time_sec"], 3),
        "window_cpi": rounded(rep["window_cpi"], 2),
        "window_mpki": rounded(rep["window_mpki"], 1),
        "full_cpi": rounded(rep["full_cpi"], 2),
        "full_mpki": rounded(rep["full_mpki"], 1),
        "cpi_rel_err": rounded(rep["cpi_rel_err"], 4),
        "mpki_rel_err": rounded(rep["mpki_rel_err"], 4),
        "cpi_rel_err_pct": rounded_pct(rep["cpi_rel_err"]),
        "mpki_rel_err_pct": rounded_pct(rep["mpki_rel_err"]),
        "score": rounded(rep["score"], 2),
    }


def _candidate_report_fields(c: Dict, n_repeats: int, include_candidate_extras: bool = False) -> Dict:
    fields = {
        "acceptable": c["acceptable"],
        "start_sec": rounded(c["start_sec"], 3),
        "end_sec": rounded(c["end_sec"], 3),
        "start_min": rounded(c["start_min"], 3),
        "end_min": rounded(c["end_min"], 3),
        "duration_sec": rounded(c["duration_sec"], 3),
        "duration_min": rounded(c["duration_min"], 3),
    }
    if n_repeats == 1:
        only_rep = next(iter(c["per_repeat"].values()))
        fields.update(
            {
                "score": rounded(c["mean_score"], 2),
                "cpi_rel_err_pct": rounded_pct(c["mean_cpi_rel_err"]),
                "mpki_rel_err_pct": rounded_pct(c["mean_mpki_rel_err"]),
                "window_cpi": rounded(only_rep["window_cpi"], 2),
                "window_mpki": rounded(only_rep["window_mpki"], 1),
            }
        )
        if include_candidate_extras:
            fields.update(
                {
                    "I_start": int(round(only_rep["I_start"])),
                    "I_end": int(round(only_rep["I_end"])),
                    "full_cpi": rounded(only_rep["full_cpi"], 2),
                    "full_mpki": rounded(only_rep["full_mpki"], 1),
                }
            )
    else:
        fields.update(
            {
                "mean_score": rounded(c["mean_score"], 2),
                "mean_cpi_rel_err_pct": rounded_pct(c["mean_cpi_rel_err"]),
                "mean_mpki_rel_err_pct": rounded_pct(c["mean_mpki_rel_err"]),
            }
        )
        if include_candidate_extras:
            fields["per_repeat_instruction_bounds"] = {
                repeat_name: {
                    "I_start": int(round(rep["I_start"])),
                    "I_end": int(round(rep["I_end"])),
                }
                for repeat_name, rep in c["per_repeat"].items()
            }
    return fields


def candidates_to_dataframe(candidates: Sequence[Dict], n_repeats: int) -> pd.DataFrame:
    rows = [_candidate_report_fields(c, n_repeats=n_repeats, include_candidate_extras=False) for c in candidates]
    return pd.DataFrame(rows)


def build_result(best: Dict, repeats: Sequence[RepeatData], top_k: int, all_candidates: Sequence[Dict]) -> Dict:
    n_repeats = len(repeats)
    result = {
        "selected_window": _candidate_report_fields(best, n_repeats=n_repeats, include_candidate_extras=True),
        "per_repeat": {},
        "top_candidates": [],
    }

    for r in repeats:
        rep = best["per_repeat"][r.name]
        result["per_repeat"][r.name] = {
            "perf_path": str(r.perf_path),
            **_per_repeat_report(rep),
        }

    for c in list(all_candidates)[:top_k]:
        result["top_candidates"].append(_candidate_report_fields(c, n_repeats=n_repeats, include_candidate_extras=False))

    return result


def print_human_summary(result: Dict) -> None:
    sel = result["selected_window"]
    score_key = "score" if "score" in sel else "mean_score"
    cpi_key = "cpi_rel_err_pct" if "cpi_rel_err_pct" in sel else "mean_cpi_rel_err_pct"
    mpki_key = "mpki_rel_err_pct" if "mpki_rel_err_pct" in sel else "mean_mpki_rel_err_pct"
    print(
        f"Selected window: start_sec={sel['start_sec']:.3f} end_sec={sel['end_sec']:.3f} "
        f"start_min={sel['start_min']:.3f} end_min={sel['end_min']:.3f} "
        f"duration_sec={sel['duration_sec']:.3f} duration_min={sel['duration_min']:.3f} "
        f"(requested={sel.get('duration_min_requested')}) acceptable={sel['acceptable']} "
        f"{score_key}={sel[score_key]:.2f} "
        f"cpi_rel_err={sel[cpi_key]:.0f}% "
        f"mpki_rel_err={sel[mpki_key]:.0f}%"
    )
    print("Per repeat:")
    for repeat_name, rep in result["per_repeat"].items():
        print(
            f"  {repeat_name}: "
            f"I_start={rep['I_start']} I_end={rep['I_end']} "
            f"t=[{rep['start_time_sec']:.3f}, {rep['end_time_sec']:.3f}] "
            f"window(CPI={rep['window_cpi']:.2f}, MPKI={rep['window_mpki']:.1f}) "
            f"full(CPI={rep['full_cpi']:.2f}, MPKI={rep['full_mpki']:.1f}) "
            f"errs(CPI={rep['cpi_rel_err_pct']:.0f}%, MPKI={rep['mpki_rel_err_pct']:.0f}%) "
            f"score={rep['score']:.2f}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the earliest/shortest interval whose CPI+MPKI matches the whole run."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Either a single perf.out file, a directory containing perf.out, or a directory containing repeat*/perf.out",
    )
    parser.add_argument(
        "--resolution-sec",
        type=int,
        default=60,
        help="Bin size in seconds for candidate windows (default: 60)",
    )
    parser.add_argument(
        "--candidate-intervals-min",
        type=int,
        nargs="+",
        default=DEFAULT_CANDIDATE_INTERVALS_MIN,
        help=f"Candidate interval lengths in minutes (default: {DEFAULT_CANDIDATE_INTERVALS_MIN})",
    )
    parser.add_argument(
        "--cpi-tol",
        type=float,
        default=0.05,
        help="Mean relative CPI error threshold for an acceptable interval (default: 0.05)",
    )
    parser.add_argument(
        "--mpki-tol",
        type=float,
        default=0.10,
        help="Mean relative MPKI error threshold for an acceptable interval (default: 0.10)",
    )
    parser.add_argument(
        "--skiprows",
        type=int,
        default=2,
        help="Skiprows passed to readPerfIntervalFile (kept for compatibility; currently readPerfIntervalFile default is used).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top candidates to include in the JSON output (default: 10)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write full JSON results.",
    )
    parser.add_argument(
        "--output-candidates-csv",
        type=Path,
        default=None,
        help="Optional path to write all candidate windows as CSV.",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="Indent to use when printing/writing JSON (default: 2)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    perf_files = find_perf_files(args.path)
    repeats = [make_repeat_data(p, resolution_sec=args.resolution_sec, skiprows=args.skiprows) for p in perf_files]

    best, all_candidates = scan_candidates(
        repeats=repeats,
        candidate_intervals_min=args.candidate_intervals_min,
        cpi_tol=args.cpi_tol,
        mpki_tol=args.mpki_tol,
    )

    result = build_result(best, repeats, top_k=args.top_k, all_candidates=all_candidates)

    if args.output_candidates_csv is not None:
        args.output_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        candidates_to_dataframe(all_candidates, n_repeats=len(repeats)).to_csv(args.output_candidates_csv, index=False)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=args.json_indent)

    print_human_summary(result)
    print(json.dumps(result, indent=args.json_indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
