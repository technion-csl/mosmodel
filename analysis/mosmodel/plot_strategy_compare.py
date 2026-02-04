#!/usr/bin/env python3
import argparse
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_model_name(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "_")

def short_gen(name: str) -> str:
    # Customize as you like
    if name == "random_window_2m":
        return "rnd2m"
    if name == "growing_window_2m":
        return "grow2m"
    if name.startswith("sliding_window/window_"):
        w = name.split("_")[-1]
        return f"sw{w}"
    if name == "moselect":
        return "moselect"
    if name == "paper_all":
        return "paper_all"
    # fallback: compact
    return name.replace("sliding_window/", "sw/").replace("_2m", "2m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Pairs: strategy_name=path/to/test_errors.csv",
    )
    ap.add_argument("--models", nargs="+", default=["mosmodel", "poly1", "poly2", "poly3"])
    args = ap.parse_args()

    models = [normalize_model_name(m) for m in args.models]
    rows = []
    strategies_in_order: List[str] = []

    for item in args.inputs:
        strat, path = item.split("=", 1)
        if strat not in strategies_in_order:
           strategies_in_order.append(strat)
        df = pd.read_csv(path)

        # Wide format expected: columns like poly1_error, poly2_error, poly3_error, mosmodel_error
        for m in models:
            col = f"{m}_error"
            if col not in df.columns:
                raise SystemExit(f"{path}: missing column {col}. Columns={list(df.columns)}")
            max_abs = df[col].abs().max()
            max_abs *= 100.0
            rows.append({"strategy": strat, "model": m, "max_error": float(max_abs)})

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No rows collected. Check inputs and --models.")

    # One value per (strategy, model)
    out = out.groupby(["strategy", "model"], as_index=False)["max_error"].max()

    order = strategies_in_order
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)

    x = list(range(len(order)))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")

    for m in models:
        ys = []
        for s in order:
            v = out[(out["strategy"] == s) & (out["model"] == m)]["max_error"]
            ys.append(float(v.iloc[0]) if len(v) else float("nan"))
        ax.plot(x, ys, marker="o", label=m)

    ax.set_ylabel("Max absolute error [%]")
 
    ax.legend()
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)

if __name__ == "__main__":
    main()
