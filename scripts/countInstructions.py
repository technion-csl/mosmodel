#! /usr/bin/env python3
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments_root")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat index (default: 1)")
    parser.add_argument("--perf_filename", default="perf.out", help="perf output filename (default: perf.out)")
    args = parser.parse_args()

    perf_file = f"{args.experiments_root}/repeat{args.repeat}/{args.perf_filename}"
    df = read_interval_perf_out(perf_file)

    instr = df.loc[df["counter_name"] == "instructions", "counter_value"].sum(min_count=1)
    if pd.isna(instr):
        raise SystemExit(f"Error: could not find numeric 'instructions' samples in {perf_file}")

    print(int(round(float(instr))))

if __name__ == "__main__":
    main()
