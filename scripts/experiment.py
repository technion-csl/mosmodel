#! /usr/bin/env python3

import pandas as pd
import numpy as np


def readSingleFile(file_name, metrics_column=0, stats_column=1):
    """Read a 2-column CSV (metric,value) into a DataFrame with index=metric and column 'stats'.

    This is still used for time.out (and any other single-summary files).
    """
    try:
        metrics, stats = np.loadtxt(
            file_name,
            delimiter=",",
            dtype=str,
            unpack=True,
            usecols=[metrics_column, stats_column],
        )
        df = pd.DataFrame({"stats": stats}, index=metrics)
        df["stats"] = pd.to_numeric(df["stats"], errors="coerce")
    except IOError:
        return None
    except Exception:
        raise ValueError("Could not read the CSV file: " + file_name)
    return df


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


class Experiment:
    def __init__(self, layout, experiments_root):
        self._layout = layout
        self._experiments_root = experiments_root

    def _experiment_dir(self, repeat):
        return f"{self._experiments_root}/{self._layout}/repeat{repeat}"

    def collect(self, repeat):
        """Collect a single-row summary for this repeat.

        perf.out is assumed to be generated with --interval-print=1000.
        We aggregate counters across time by summing per-interval counts.
        """
        experiment_dir = self._experiment_dir(repeat)

        perf_file_name = f"{experiment_dir}/perf.out"
        perf_ts = readPerfIntervalFile(perf_file_name)
        if perf_ts is None:
            return None

        totals = perf_ts.sum(axis=0, skipna=True)
        perf_df = pd.DataFrame({"stats": totals.values}, index=totals.index)

        time_file_name = f"{experiment_dir}/time.out"
        time_df = readSingleFile(time_file_name)

        df = pd.concat([perf_df, time_df])
        df = df.transpose()
        return df

    def collect_interpolated(self, repeat, instruction_count, perf_out_name="perf.out"):
        """Sample metrics at the moment a target instruction count is reached.

        We parse perf.out (interval CSV), convert per-interval counts into
        cumulative totals via cumsum, then interpolate (linearly) between the
        two bracketing samples where cumulative 'instructions' crosses the target.

        Returns:
            A single-row DataFrame (like collect()).
        """
        experiment_dir = self._experiment_dir(repeat)
        perf_path = f"{experiment_dir}/{perf_out_name}"

        ts = readPerfIntervalFile(perf_path)
        if ts is None:
            return None

        if "instructions" not in ts.columns:
            raise ValueError(f"{perf_path}: missing required counter 'instructions'")

        # Per-interval -> cumulative
        cum = ts.cumsum(axis=0).interpolate(limit_direction="both")

        instr = cum["instructions"].to_numpy(dtype=float)
        t = cum.index.to_numpy(dtype=float)

        target = float(instruction_count)
        instr_min = float(np.nanmin(instr))
        instr_max = float(np.nanmax(instr))

        if target < instr_min:
            raise ValueError(
                f"instruction_count={target} out of range for {perf_path} "
                f"(min={instr_min})"
            )
        
        if target >= instr_max:
            # allow "didn't reach target" => take full run
            sampled = cum.iloc[-1]
        else:
        # Find bracket
            idx = int(np.searchsorted(instr, target, side="left"))
            if idx == 0:
                sampled = cum.iloc[0]
            else:
                i0, i1 = idx - 1, idx
                instr0, instr1 = instr[i0], instr[i1]

                if instr1 == instr0:
                    sampled = cum.iloc[i1]
                else:
                    w = (target - instr0) / (instr1 - instr0)
                    sampled = (1.0 - w) * cum.iloc[i0] + w * cum.iloc[i1]

        out = sampled.to_frame().T
        out.index = ["stats"]
        return out

    def __repr__(self):
        return "experiment with " + str(self.__dict__)
