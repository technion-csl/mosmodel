#! /usr/bin/env python3

import numpy as np
import pandas as pd


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

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["counter_value"] = pd.to_numeric(df["counter_value"], errors="coerce")
    df["counter_name"] = df["counter_name"].astype(str).str.strip()

    df = df.dropna(subset=["time", "counter_name"]).sort_values("time")
    wide = df.pivot(index="time", columns="counter_name", values="counter_value")
    wide.columns = [str(c).replace(":u", "") for c in wide.columns]
    wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide


class Experiment:
    def __init__(self, layout, experiments_root):
        self._layout = layout
        self._experiments_root = experiments_root

    def _layout_dir(self):
        return f"{self._experiments_root}/{self._layout}"

    def _experiment_dir(self, repeat):
        return f"{self._layout_dir()}/repeat{repeat}"

    def _read_perf_cumulative(self, repeat, perf_out_name="perf.out"):
        experiment_dir = self._experiment_dir(repeat)
        perf_path = f"{experiment_dir}/{perf_out_name}"

        ts = readPerfIntervalFile(perf_path)
        if ts is None:
            return None, perf_path

        if "instructions" not in ts.columns:
            raise ValueError(f"{perf_path}: missing required counter 'instructions'")

        cum = ts.cumsum(axis=0).interpolate(limit_direction="both")
        instr = cum["instructions"].to_numpy(dtype=float)
        t = cum.index.to_numpy(dtype=float)
        return {"cum": cum, "instr": instr, "time": t, "perf_path": perf_path}, perf_path

    @staticmethod
    def _zero_snapshot(columns):
        return pd.Series(np.zeros(len(columns), dtype=float), index=columns)

    def _sample_cumulative_at_instruction(self, cum_ctx, instruction_count):
        """Return cumulative counters and elapsed time at a target retired-instruction count.

        Supports:
          - target <= 0 => zero snapshot at time 0
          - 0 < target < first cumulative sample => linear interpolation from zero state
          - target >= full-run instructions => last cumulative sample
        """
        cum = cum_ctx["cum"]
        instr = cum_ctx["instr"]
        t = cum_ctx["time"]
        target = float(instruction_count)

        if target <= 0:
            return self._zero_snapshot(cum.columns), 0.0

        instr_max = float(np.nanmax(instr))
        if target >= instr_max:
            return cum.iloc[-1].copy(), float(t[-1])

        idx = int(np.searchsorted(instr, target, side="left"))

        if idx == 0:
            instr1 = instr[0]
            if not np.isfinite(instr1) or instr1 <= 0:
                return cum.iloc[0].copy(), float(t[0])
            w = target / instr1
            sampled = w * cum.iloc[0]
            sampled_time = w * float(t[0])
            return sampled, sampled_time

        i0, i1 = idx - 1, idx
        instr0, instr1 = instr[i0], instr[i1]

        if instr1 == instr0:
            return cum.iloc[i1].copy(), float(t[i1])

        w = (target - instr0) / (instr1 - instr0)
        sampled = (1.0 - w) * cum.iloc[i0] + w * cum.iloc[i1]
        sampled_time = (1.0 - w) * float(t[i0]) + w * float(t[i1])
        return sampled, sampled_time

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

    def collect_instruction_interval(self, repeat, I_start, I_end, perf_out_name="perf.out"):
        """Collect metrics over the cumulative-instruction interval [I_start, I_end]."""
        ctx, _ = self._read_perf_cumulative(repeat, perf_out_name=perf_out_name)
        if ctx is None:
            return None

        I_start = float(I_start)
        I_end = float(I_end)
        if I_end < I_start:
            raise ValueError(f"Invalid interval: I_end ({I_end}) < I_start ({I_start})")

        start_sample, start_time = self._sample_cumulative_at_instruction(ctx, I_start)
        end_sample, end_time = self._sample_cumulative_at_instruction(ctx, I_end)

        delta = end_sample - start_sample
        out = delta.to_frame().T
        out.index = ["stats"]
        out["seconds-elapsed"] = float(end_time - start_time)
        out["I_start"] = float(I_start)
        out["I_end"] = float(I_end)
        out["sampled_start_sec"] = float(start_time)
        out["sampled_end_sec"] = float(end_time)
        return out

    def collect_interval(self, repeat, I_start=None, I_end=None, perf_out_name="perf.out"):
        """Collect either the full run or the explicit interval [I_start, I_end]."""
        if I_start is None and I_end is None:
            return self.collect(repeat)
        if I_start is None or I_end is None:
            raise ValueError("Both I_start and I_end must be provided together")
        return self.collect_instruction_interval(
            repeat,
            I_start,
            I_end,
            perf_out_name=perf_out_name,
        )

    def __repr__(self):
        return "experiment with " + str(self.__dict__)
