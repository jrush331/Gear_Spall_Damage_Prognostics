"""Flatten the per-minute PHM HDF5 files into analysis-ready tables (one set per split).

Each HDF5 file is ~1 minute of dyno logging with these groups:
    CI/<metric>        (≈60,)  condition indicators, one value per 1-s vibration window
    Context/<channel>  (≈60,)  speed, torque, temperature, accel RMS, sampled once per second
    DM4500 Data, ICM2 Data     one oil-debris particle-count reading per file (8 size bins each)
    (HF files only) CI_4s, Transforms, Transforms_4s, Vibration  -- 4-s CIs and raw waveforms

CI and Context are sampled on different clocks (0.5 s apart, and often 59 vs 60 rows), so rows are aligned
on timestamps derived from each dataset's wf_start_time / wf_start_offset / wf_increment, never on array
position.

Outputs (data/<split>/):
    lf_1s.parquet      one row per second: ids, timestamp, test_hours, ci_* and ctx_* columns
    particles.parquet  one row per file: ids, timestamp, dm4500_bin0..7, icm2_bin0..7
data/channels.csv      units and processing attributes for every column

Usage:
    .venv/bin/python scripts/flatten_hdf5.py [--split test] [--workers 8]
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
SPLITS = ["train", "test", "validation"]
PER_SECOND = {"CI": "ci", "Context": "ctx"}
PARTICLES = {"DM4500 Data": "dm4500", "ICM2 Data": "icm2"}


def col_name(prefix, name):
    return f"{prefix}_" + re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def attr(ds, key, default=None):
    v = ds.attrs.get(key, default)
    return v.decode() if isinstance(v, bytes) else v


def sample_times(ds):
    """Absolute timestamp of each sample; for windowed CIs, the start of each window."""
    t0 = pd.Timestamp(attr(ds, "wf_start_time")).tz_localize(None)
    inc = float(attr(ds, "wf_increment", 1.0))
    # CIs carry offset=1 (value stamped at end of its 1-s window); shift back so row i = window [t0+i, t0+i+1)
    start = float(attr(ds, "wf_start_offset", 0.0)) - (
        inc if ds.parent.name.startswith("/CI") else 0.0
    )
    return t0 + pd.to_timedelta(start + inc * np.arange(ds.shape[0]), unit="s")


def read_file(args):
    path, row = args
    ids = dict(
        experiment=row["experiment"],
        run=row["run"],
        run_num=int(row["run"].split("-")[1]),
        file=path.name,
    )
    m = re.search(r"Dyno (Gear\d+Run\d+)_(\d+)(?:_chunk_(\d+))?", row["inner_path"])
    ids.update(session=m.group(1), file_seq=int(m.group(2)), chunk=int(m.group(3) or 0))
    series, particles = [], dict(ids)
    try:
        with h5py.File(path) as f:
            for group, prefix in PER_SECOND.items():
                for name, ds in f.get(group, {}).items():
                    if ds.shape[0]:
                        series.append(
                            pd.Series(
                                ds[()],
                                index=sample_times(ds),
                                name=col_name(prefix, name),
                            )
                        )
            for group, prefix in PARTICLES.items():
                for name, ds in f.get(group, {}).items():
                    particles[col_name(prefix, name.split(" ", 1)[1])] = (
                        int(ds[0]) if ds.shape[0] else None
                    )
                    particles.setdefault(
                        "timestamp", sample_times(ds)[0] if ds.shape[0] else pd.NaT
                    )
    except OSError as e:  # truncated / zero-byte files exist in the source
        return None, None, f"{path.name}: {e}"
    if not series:
        return None, particles, None
    # align every channel on a 1-s grid anchored at the earliest window start in this file
    t0 = min(s.index[0] for s in series)
    for s in series:
        s.index = ((s.index - t0) / pd.Timedelta(seconds=1)).astype(int)
        if s.index.has_duplicates:
            s.drop(s.index[s.index.duplicated()], inplace=True)
    df = pd.concat(series, axis=1).sort_index()
    df.insert(0, "timestamp", t0 + pd.to_timedelta(df.index, unit="s"))
    df.insert(0, "second", df.index)
    for k, v in reversed(ids.items()):
        df.insert(0, k, v)
    return df.reset_index(drop=True), particles, None


def channel_info(sample_path):
    rows = []
    with h5py.File(sample_path) as f:
        for group, prefix in {**PER_SECOND, **PARTICLES}.items():
            for name, ds in f.get(group, {}).items():
                a = {k: attr(ds, k) for k in ds.attrs if not k.startswith("wf_")}
                unit = a.pop("unit_string", None) or a.pop("NI_UnitDescription", None)
                a.pop("NI_UnitDescription", None)
                c = col_name(
                    prefix, name if group in PER_SECOND else name.split(" ", 1)[1]
                )
                rows.append(
                    dict(column=c, hdf5_path=f"{group}/{name}", unit=unit, attrs=str(a))
                )
    return pd.DataFrame(rows)


def flatten(split, workers):
    m = pd.read_csv(DATA / split / "manifest.csv")
    m = m[m.modality == "lf"]
    jobs = [(DATA / split / r["file"], r) for r in m.to_dict("records")]
    frames, parts, errors = [], [], []
    with ProcessPoolExecutor(workers) as pool:
        for df, p, err in pool.map(read_file, jobs, chunksize=64):
            if df is not None:
                frames.append(df)
            if p is not None and "timestamp" in p:
                parts.append(p)
            if err:
                errors.append(err)
    lf = pd.concat(frames, ignore_index=True)
    # order by official run label then wall-clock time. test_hours = cumulative seconds with CI data within an
    # experiment; this reproduces the submission templates' run-end times to within a few minutes
    lf = lf.sort_values(
        ["experiment", "run_num", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    has_ci = lf.filter(like="ci_").notna().any(axis=1)
    lf.insert(
        lf.columns.get_loc("timestamp") + 1,
        "test_hours",
        has_ci.groupby(lf.experiment).cumsum() / 3600,
    )
    pt = (
        pd.DataFrame(parts)
        .sort_values(["experiment", "run_num", "timestamp"])
        .reset_index(drop=True)
    )
    lf.to_parquet(DATA / split / "lf_1s.parquet", index=False)
    pt.to_parquet(DATA / split / "particles.parquet", index=False)
    print(
        f"{split}: lf_1s {lf.shape}, particles {pt.shape}, unreadable files {len(errors)}"
    )
    for e in errors:
        print("   ", e)
    return lf, lf.loc[
        lf.filter(like="ci_").notna().all(axis=1)
        & lf.filter(like="ctx_").notna().all(axis=1),
        "file",
    ]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=SPLITS, action="append")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for split in a.split or SPLITS:
        _, complete = flatten(split, a.workers)
    channel_info(DATA / split / "lf" / complete.iloc[0]).to_csv(
        DATA / "channels.csv", index=False
    )


if __name__ == "__main__":
    main()
