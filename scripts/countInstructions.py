#! /usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def read_interval_perf_out(perf_file: str) -> pd.DataFrame:
    # Typical perf stat CSV interval output has 2 header lines, then rows like:
    #   <time>,<count>,<unit>,<event>
    df = pd.read_csv(
        perf_file,
        skiprows=[0, 1],
        header=None,
        usecols=[0, 1, 3],
        names=["time", "counter_value", "counter_name"],
        na_values=["<not counted>"],
        comment="#",
        engine="python",
    ).dropna(subset=["counter_name"])

    # Normalize event name: strip whitespace and drop ':u' / ':k' qualifiers etc.
    df["counter_name"] = (
        df["counter_name"].astype(str).str.strip().str.split(":").str[0]
    )

    # Numbers sometimes come with spaces; coerce non-numeric to NaN
    df["counter_value"] = df["counter_value"].astype(str).str.replace(" ", "", regex=False)
    df["counter_value"] = pd.to_numeric(df["counter_value"], errors="coerce")
    return df


def load_instruction_interval_bounds(experiments_root: str, repeat: int, instruction_interval_json: str):
    json_path = Path(instruction_interval_json)
    if not json_path.exists():
        json_path = Path(experiments_root) / instruction_interval_json
    if not json_path.exists():
        raise SystemExit(f"Error: interval JSON was not found: {instruction_interval_json}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        raise SystemExit(f"Error: could not read interval JSON {json_path}: {e}")

    per_repeat = payload.get("per_repeat")
    if not isinstance(per_repeat, dict):
        raise SystemExit(f"Error: {json_path} is missing 'per_repeat'")

    repeat_key = f"repeat{repeat}"
    rep = per_repeat.get(repeat_key)
    if not isinstance(rep, dict):
        raise SystemExit(f"Error: {json_path} is missing entry for {repeat_key}")

    if "I_start" not in rep or "I_end" not in rep:
        raise SystemExit(f"Error: {json_path} {repeat_key} is missing I_start/I_end")

    I_start = float(rep["I_start"])
    I_end = float(rep["I_end"])
    if I_end < I_start:
        raise SystemExit(f"Error: invalid interval in {json_path} for {repeat_key}: I_end < I_start")
    return I_start, I_end


def full_run_instruction_bounds(experiments_root: str, repeat: int, perf_filename: str):
    perf_file = f"{experiments_root}/repeat{repeat}/{perf_filename}"
    df = read_interval_perf_out(perf_file)

    instr = df.loc[df["counter_name"] == "instructions", "counter_value"].sum(min_count=1)
    if pd.isna(instr):
        raise SystemExit(f"Error: could not find numeric 'instructions' samples in {perf_file}")

    return 0.0, float(instr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments_root")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat index (default: 1)")
    parser.add_argument("--perf_filename", default="perf.out", help="perf output filename (default: perf.out)")
    parser.add_argument(
        "--instruction_interval_json",
        default=None,
        help="optional JSON file with per_repeat I_start/I_end. If provided, prints I_start,I_end; otherwise prints 0,full_run_instructions",
    )
    args = parser.parse_args()

    if args.instruction_interval_json is not None:
        I_start, I_end = load_instruction_interval_bounds(
            args.experiments_root,
            args.repeat,
            args.instruction_interval_json,
        )
    else:
        I_start, I_end = full_run_instruction_bounds(
            args.experiments_root,
            args.repeat,
            args.perf_filename,
        )

    print(f"{int(round(float(I_start)))},{int(round(float(I_end)))}")


if __name__ == "__main__":
    main()
