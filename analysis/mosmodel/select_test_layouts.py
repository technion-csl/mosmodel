#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        if r.fieldnames is None:
            raise SystemExit(f"{path}: missing header")
        return list(r.fieldnames), rows

def sample_unique(rows: List[Dict[str, str]], k: int, rng: random.Random) -> List[Dict[str, str]]:
    if k <= 0:
        return []
        
    # fallback: unique rows by full row tuple
    uniq_map = {}
    for row in rows:
        uniq_map.setdefault(tuple(row.items()), row)
    uniq_rows = list(uniq_map.values())
    if k > len(uniq_rows):
        k = len(uniq_rows)
    return rng.sample(uniq_rows, k)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input mean.csv files (results/*/mean.csv)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_total", type=int, required=True, help="Total layouts to select")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mode", choices=["uniform", "stratified", "moselect_plus_uniform"], default="stratified",
                help="uniform=sample from union; stratified=equal per input; moselect_plus_uniform=take all moselect then fill uniformly from others")

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

    max_unique = len({tuple(sorted(r.items())) for _, _, rows in all_inputs for r in rows})
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
        union_rows: List[Dict[str, str]] = []
        for _, _, rows in all_inputs:
            union_rows.extend(rows)
        selected = sample_unique(union_rows, args.num_total, rng)

    else:  # moselect_plus_uniform
        # 1) take all moselect rows
        moselect_rows: List[Dict[str, str]] = []
        other_rows: List[Dict[str, str]] = []

        for p, _, rows in all_inputs:
            # adjust the predicate to match your naming conventions
            if "moselect" in str(p):
                moselect_rows.extend(rows)
            else:
                other_rows.extend(rows)

        # de-dup moselect rows by full signature
        seen = set()
        for r in moselect_rows:
            sig = tuple(sorted(r.items()))
            if sig in seen:
                continue
            selected.append(r)
            seen.add(sig)

        # 2) fill the rest uniformly from "others"
        remaining = max(0, args.num_total - len(selected))
        if remaining > 0:
            extra = sample_unique(other_rows, remaining, rng)
            for r in extra:
                sig = tuple(sorted(r.items()))
                if sig in seen:
                    continue
                selected.append(r)
                seen.add(sig)
                if len(selected) >= args.num_total:
                    break

        # If moselect already exceeds num_total, truncate
        selected = selected[:args.num_total]


    # De-duplicate again across inputs (in case same layout appears in multiple sources)

    # If dedup reduced count, top-up (uniformly) from union
    if len(selected) < args.num_total:
        union_rows = []
        for _, _, rows in all_inputs:
            union_rows.extend(rows)
        need = args.num_total - len(selected)
        extra = sample_unique(union_rows, need, rng)

        seen = {tuple(sorted(r.items())) for r in selected}
        for r in extra:
            sig = tuple(sorted(r.items()))
            if sig in seen:
                continue
            selected.append(r)
            seen.add(sig)
            if len(selected) >= args.num_total:
                break


    # If still not enough, just write what we have
    selected = selected[:args.num_total]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers0)
        w.writeheader()
        w.writerows(selected)

if __name__ == "__main__":
    main()

