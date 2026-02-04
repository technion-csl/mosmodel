#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict, Tuple

def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        if r.fieldnames is None:
            raise SystemExit(f"{path}: missing header")
        return list(r.fieldnames), rows

def row_sig(row: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(row.items()))

def unique_preserve_order(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for r in rows:
        sig = row_sig(r)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(r)
    return out

def sample_unique(rows: List[Dict[str, str]], k: int, rng: random.Random) -> List[Dict[str, str]]:
    if k <= 0:
        return []
    uniq_rows = unique_preserve_order(rows)
    if k > len(uniq_rows):
        k = len(uniq_rows)
    return rng.sample(uniq_rows, k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input mean.csv files (results/*/mean.csv)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--mode",
        choices=["uniform", "stratified", "moselect_plus_uniform"],
        default="stratified",
        help="uniform=sample from union; stratified=equal per input; "
             "moselect_plus_uniform=take all moselect then fill uniformly from others",
    )

    # NEW: either --num_total or --all_layouts
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--num_total", type=int, help="Total layouts to select")
    g.add_argument("--all_layouts", action="store_true", help="Select all unique layouts from all inputs")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    in_paths = [Path(p) for p in args.inputs]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all inputs + sanity-check headers
    headers0, rows0 = read_csv(in_paths[0])
    all_inputs = [(in_paths[0], headers0, rows0)]
    for p in in_paths[1:]:
        hdr, rows = read_csv(p)
        if hdr != headers0:
            raise SystemExit(f"Header mismatch between {in_paths[0]} and {p}")
        all_inputs.append((p, hdr, rows))

    # Union of all rows
    union_rows: List[Dict[str, str]] = []
    for _, _, rows in all_inputs:
        union_rows.extend(rows)

    # NEW: all layouts mode
    if args.all_layouts:
        selected = unique_preserve_order(union_rows)
        print(f"Selecting ALL unique layouts: {len(selected)}")
    else:
        max_unique = len({row_sig(r) for r in union_rows})
        args.num_total = min(args.num_total, max_unique)
        print(f"Selecting {args.num_total} unique layouts (max available: {max_unique})")

        selected: List[Dict[str, str]] = []

        if args.mode == "stratified":
            k = len(all_inputs)
            base = args.num_total // k
            rem = args.num_total % k
            per = [base + (1 if i < rem else 0) for i in range(k)]
            for (p, _, rows), take in zip(all_inputs, per):
                selected.extend(sample_unique(rows, take, rng))

        elif args.mode == "uniform":
            selected = sample_unique(union_rows, args.num_total, rng)

        else:  # moselect_plus_uniform
            moselect_rows: List[Dict[str, str]] = []
            other_rows: List[Dict[str, str]] = []
            for p, _, rows in all_inputs:
                if "moselect" in str(p):
                    moselect_rows.extend(rows)
                else:
                    other_rows.extend(rows)

            selected = unique_preserve_order(moselect_rows)

            remaining = max(0, args.num_total - len(selected))
            if remaining > 0:
                extra = sample_unique(other_rows, remaining, rng)
                # top-up while preserving uniqueness against already selected
                seen = {row_sig(r) for r in selected}
                for r in extra:
                    sig = row_sig(r)
                    if sig in seen:
                        continue
                    selected.append(r)
                    seen.add(sig)
                    if len(selected) >= args.num_total:
                        break

            selected = selected[:args.num_total]

        # Final de-dup + top-up from union if needed
        selected = unique_preserve_order(selected)
        if len(selected) < args.num_total:
            need = args.num_total - len(selected)
            extra = sample_unique(union_rows, need, rng)
            seen = {row_sig(r) for r in selected}
            for r in extra:
                sig = row_sig(r)
                if sig in seen:
                    continue
                selected.append(r)
                seen.add(sig)
                if len(selected) >= args.num_total:
                    break

        selected = selected[:args.num_total]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers0)
        w.writeheader()
        w.writerows(selected)

if __name__ == "__main__":
    main()
