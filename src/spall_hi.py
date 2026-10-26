"""Unsupervised gear-damage health index (HI) for the PHM 2026 data challenge.

No damage labels are used to fit anything. Every model is fit on *healthy* data only
(the first hours of each training gear), and damage is scored as distance from that
healthy state. The submitted HI is fixed in advance (tooth) or chosen with label-free
criteria (label_free_table); the per-tooth condition labels (<data>/labels/condition_labels.csv)
are used only to report how candidates score under the challenge's rule, unless --select labels is passed.

Pipeline
  1. 1-s rows -> steady-state filter -> 1-min medians            (load_minutes)
  2. log|CI| -> regress out temperature/torque per gear          (condition_correct)
  3. standardise per gear on its own healthy baseline window     (condition_correct)
  4. fit healthy-state models on pooled training baselines       (HealthyModel)
  5. HI per minute -> value at each run end, made monotone       (run_end_scores)

Candidates: all-CI novelty (maha, spe, zpos, zmax) and a directional tooth-damage HI on
FM4/M6A/M8A/NA4 (tooth, tooth_maha). The all-CI scores are dominated by the level shifts
that follow every rig restart; the tooth HIs stay flat on healthy runs and rise at spall.

Usage
  python models/spall_hi.py                 # evaluate candidates, write submissions to models/out/
  python models/spall_hi.py --data /path    # folder holding <split>/lf_1s.parquet, particles.parquet, labels/
  python models/spall_hi.py --select label_free   # submit the HI with the best label-free fitness
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

SPLITS = {"train": ["Exp-A", "Exp-B", "Exp-C", "Exp-F"], "validation": ["Exp-G"], "test": ["Exp-D", "Exp-E"]}
# Gear-mesh CIs with a damage interpretation. Excluded: *star variants (baselined on another rig
# run), fleet_wtmade (trained on other experiments), raw rms/ptp/v_peak (load-dominated).
FEATS = [
    "ci_fm4",
    "ci_m6a",
    "ci_m8a",
    "ci_na4",
    "ci_e_ratio",
    "ci_kurtosis_tsa",
    "ci_rms_tsa",
    "ci_alr",
    "ci_fm0",
    "ci_crest_tsa",
    "ci_kurtosis_e_op",
    "ci_rms_e_op",
]
# Localised-tooth-damage CIs (sidebands / residual impulsiveness around mesh). In EDA these stay near 0 on
# healthy runs, while the energy and global CIs above shift by tens of sigma after each rig restart.
TOOTH = ["ci_fm4", "ci_m6a", "ci_m8a", "ci_na4"]
COND = ["ctx_temperature", "ctx_pau_torque"]
BREAK_IN_H = 0.5  # skip the first half hour of every gear (run-in transients)
BASE_H = 6.0  # healthy window = [BREAK_IN_H, BASE_H]: inside run 1 of every gear
TAIL_MIN = 60  # run-end value = TAIL_Q quantile of HI over the last TAIL_MIN steady minutes of the run
TAIL_Q = 0.9  # upper quantile: post-spall spikes are short, a median over a longer tail misses them
ALARM_Q = 0.99  # label-free false-alarm threshold: this quantile of the HI on the fit gears' baseline minutes
HI_DEFAULT = "tooth"  # submitted HI, fixed in advance; see --select for the alternatives
MAX_DAMAGE = 3.0  # condition-label damage scale: score 4 (severe spalling) - 1 (healthy)
DEBRIS_BINS =[f"dm4500_bin_{i}" for i in range(2, 8)]  # larger ferrous particles

# Photo-based damage per (gear, inspection-after-run) from the by-eye labels in eda/scripts/compute.py:
# initiation = 1, spall = 2, summed over teeth. Kept as a secondary check (eval T2); the primary reference is the
# per-tooth condition labels (label_damage). Evaluation only; never used for fitting.
EYE = {
    ("Exp-A", 16, 4): 1,
    ("Exp-A", 28, 4): 1,
    ("Exp-B", 11, 6): 1,
    ("Exp-B", 14, 6): 1,
    ("Exp-F", 9, 7): 1,
    ("Exp-F", 12, 7): 1,
    ("Exp-F", 13, 7): 1,
}
EYE.update(
    {
        k: 2
        for k in [
            ("Exp-A", 16, 5),
            ("Exp-A", 24, 5),
            ("Exp-A", 28, 5),
            ("Exp-B", 11, 7),
            ("Exp-B", 14, 7),
            ("Exp-C", 6, 12),
            ("Exp-C", 7, 12),
            ("Exp-F", 9, 8),
            ("Exp-F", 12, 8),
            ("Exp-F", 13, 8),
            ("Exp-F", 20, 8),
            ("Exp-F", 19, 8),
            ("Exp-F", 18, 6),
            ("Exp-F", 18, 7),
            ("Exp-F", 18, 8),
        ]
    }
)


# ---------------------------------------------------------------- data
def load_minutes(data: Path, split: str, steady: bool = True) -> pd.DataFrame:
    lf = pd.read_parquet(data / split / "lf_1s.parquet")
    ci = [c for c in lf if c.startswith(("ci_", "ctx_"))]
    # Exp-A Run-2 is a byte copy of Run-1 at the source: keep the hour axis, drop the values
    lf.loc[(lf.experiment == "Exp-A") & (lf.run_num == 2), ci] = np.nan
    if steady:
        nom_t = lf.groupby("experiment").ctx_pau_torque.transform("median")
        keep = (lf.ctx_pau_speed.sub(1400).abs() < 20) & (lf.ctx_pau_torque.sub(nom_t).abs() < 0.05 * nom_t)
        lf = lf[keep]
    lf = lf.assign(minute=(lf.test_hours * 60).astype(int))
    m = lf.groupby(["experiment", "run_num", "minute"])[FEATS + COND].median().reset_index()
    m["hours"] = (m.minute + 0.5) / 60
    return m.dropna(subset=FEATS + COND)


def load_debris(data: Path, split: str) -> pd.DataFrame:
    """Large-particle count rate per run (particles/hour), from the cumulative DM4500 counters."""
    p = pd.read_parquet(data / split / "particles.parquet").sort_values(["experiment", "timestamp"])
    p = p[~((p.experiment == "Exp-A") & (p.run_num == 2))]
    p["dm"] = p[DEBRIS_BINS].sum(axis=1)
    g = p.groupby(["experiment", "run_num"]).agg(
        c0=("dm", "first"), c1=("dm", "last"), t0=("timestamp", "first"), t1=("timestamp", "last")
    )
    hrs = (g.t1 - g.t0).dt.total_seconds() / 3600
    return ((g.c1 - g.c0) / hrs.clip(lower=0.25)).rename("debris_rate").reset_index()


def run_ends(data: Path, split: str) -> pd.DataFrame:
    lf = pd.read_parquet(data / split / "lf_1s.parquet", columns=["experiment", "run_num", "test_hours"])
    return lf.groupby(["experiment", "run_num"]).test_hours.max().rename("end_h").reset_index()


# ---------------------------------------------------------------- features
def condition_correct(
    m: pd.DataFrame,
    base_h: float = BASE_H,
    break_in_h: float = BREAK_IN_H,
    correct: bool = True,
    per_gear: bool = True,
) -> pd.DataFrame:
    """log|CI| with temperature/torque regressed out and scaled on each gear's own healthy window.

    Gears run at different loads (Exp-A at 120 ft-lbf, the rest at 100) and every gear has its own
    healthy vibration signature, so each gear is referenced to itself; the pooled models then only
    see departures from that gear's normal. correct=False skips the regression and per_gear=False
    uses one reference pooled over all gears in `m` (both for ablation only).
    """

    def prep(g):
        X = np.log(g[FEATS].abs().to_numpy() + 1e-9)
        base = ((g.hours > break_in_h) & (g.hours < base_h)).to_numpy()
        return X, g[COND].to_numpy(), base

    def reference(X, C, base):
        mu, sd = C[base].mean(0), C[base].std(0) + 1e-9
        T = np.c_[np.ones(len(C)), (C - mu) / sd]
        beta = np.linalg.lstsq(T[base], X[base], rcond=None)[0] if correct else np.zeros((T.shape[1], X.shape[1]))
        R = X - T @ beta
        med = np.median(R[base], 0)
        mad = np.median(np.abs(R[base] - med), 0) * 1.4826
        return mu, sd, beta, med, np.where(mad > 0, mad, 1)

    def apply(X, C, ref):
        mu, sd, beta, med, mad = ref
        T = np.c_[np.ones(len(C)), (C - mu) / sd]
        return (X - T @ beta - med) / mad

    pooled = None if per_gear else reference(*prep(m))
    out = []
    for _, g in m.groupby("experiment"):
        X, C, base = prep(g)
        Z = apply(X, C, pooled if pooled is not None else reference(X, C, base))
        z = pd.DataFrame(Z, columns=FEATS, index=g.index)
        z[["experiment", "run_num", "hours"]] = g[["experiment", "run_num", "hours"]]
        z["base"] = base
        out.append(z)
    return pd.concat(out)


class HealthyModel:
    """Healthy-state models fit on pooled baseline minutes of the training gears only."""

    def __init__(self, tooth: list[str] = TOOTH, feats: list[str] = FEATS):
        self.tooth, self.feats = tooth, feats

    def fit(self, z: pd.DataFrame):
        B = z.loc[z.base, self.feats].to_numpy()
        self.lw = LedoitWolf().fit(B)
        self.lw_t = LedoitWolf().fit(z.loc[z.base, self.tooth].to_numpy())
        self.pca = PCA(n_components=0.95, svd_solver="full").fit(B)
        R = B - self.pca.inverse_transform(self.pca.transform(B))
        self.spe_scale = np.median((R**2).sum(1))
        return self

    def score(self, z: pd.DataFrame) -> pd.DataFrame:
        X = z[self.feats].to_numpy()
        R = X - self.pca.inverse_transform(self.pca.transform(X))
        s = z[["experiment", "run_num", "hours"]].copy()
        s["maha"] = np.sqrt(self.lw.mahalanobis(X) / len(self.feats))
        s["spe"] = np.sqrt((R**2).sum(1) / self.spe_scale)
        # directional: damage raises impulsiveness/energy CIs, so only upward departures count
        s["zpos"] = np.sqrt((np.clip(X, 0, None) ** 2).mean(1))
        s["zmax"] = np.abs(X).max(1)
        T = z[self.tooth].to_numpy()
        s["tooth"] = np.clip(T, 0, None).mean(1)
        s["tooth_maha"] = np.sqrt(self.lw_t.mahalanobis(T) / len(self.tooth)) * (T.mean(1) > 0)
        return s


# ---------------------------------------------------------------- run-end scores
def _iso(v: pd.Series) -> pd.Series:
    return pd.Series(IsotonicRegression().fit_transform(np.arange(len(v)), v.to_numpy()), index=v.index)


def run_end_scores(
    s: pd.DataFrame,
    ends: pd.DataFrame,
    cols: list[str],
    monotone: str = "iso",
    tail: int = TAIL_MIN,
    q: float = TAIL_Q,
) -> pd.DataFrame:
    """One value per run end: q-quantile of the last `tail` minutes, carried forward, then made monotone.

    Spalls do not heal, so the reported damage never goes down. "iso" fits a non-decreasing step
    function per gear (a one-run spike is averaged with its neighbours); "cummax" locks spikes in.
    Runs with no usable minutes (Exp-A Run-2) inherit the previous value.
    """
    tail = (
        s.sort_values("hours")
        .groupby(["experiment", "run_num"])
        .tail(tail)
        .groupby(["experiment", "run_num"])[cols]
        .quantile(q)
        .reset_index()
    )
    r = ends.merge(tail, on=["experiment", "run_num"], how="left").sort_values(["experiment", "run_num"])
    r[cols] = r.groupby("experiment")[cols].ffill().fillna(0)
    if monotone == "cummax":
        r[cols] = r.groupby("experiment")[cols].cummax()
    elif monotone == "iso":
        r[cols] = r.groupby("experiment")[cols].transform(_iso)
    return r


# ---------------------------------------------------------------- evaluation
def photo_damage(ends: pd.DataFrame) -> pd.Series:
    """Photo damage after each training run (sum of by-eye tooth scores); evaluation only."""
    d = pd.Series(EYE).rename_axis(["experiment", "tooth", "run_num"]).groupby(level=[0, 2]).sum()
    idx = pd.MultiIndex.from_frame(ends[["experiment", "run_num"]])
    return d.reindex(idx, fill_value=0).to_numpy(dtype=float)


def label_damage(ends: pd.DataFrame, labels: Path) -> np.ndarray:
    """Mean tooth damage after each run from the condition labels: mean over teeth of (score - 1), where 1 = healthy,
    2 = spall onset, 3 = spalling, 4 = severe spalling. Evaluation only; never used for fitting."""
    d = pd.read_csv(labels).assign(ex=lambda x: x.score - 1).groupby(["experiment", "run_num"]).ex.mean()
    return d.reindex(pd.MultiIndex.from_frame(ends[["experiment", "run_num"]])).to_numpy(dtype=float)


def challenge_score(pred, truth, groups=None, scale: float | None = None) -> dict:
    """Approximate the organisers' metric: MSE after the best monotone (isotonic) map of pred onto truth.

    pooled: one map shared by all gears, so the HI scale has to agree across gears.
    per_gear: a separate map per gear (only trajectory shape matters). Truth is divided by `scale` (MAX_DAMAGE for
    the condition labels, so errors are comparable across label sets) or, if None, by its own maximum.
    """
    truth = np.asarray(truth, float)
    pred, truth = np.asarray(pred, float), truth / (scale or max(np.max(truth), 1e-9))
    pooled = IsotonicRegression(out_of_bounds="clip").fit(pred, truth).predict(pred)
    out = {"mse_pooled": float(np.mean((pooled - truth) ** 2)), "spearman": float(spearmanr(pred, truth)[0])}
    if groups is not None:
        err = np.zeros_like(truth)
        for g in np.unique(groups):
            k = groups == g
            err[k] = IsotonicRegression(out_of_bounds="clip").fit(pred[k], truth[k]).predict(pred[k]) - truth[k]
        out["mse_per_gear"] = float(np.mean(err**2))
    return out


def logo_scores(z_tr: pd.DataFrame, alarm_q: float = ALARM_Q) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-gear-out per-minute HIs: healthy models are refit without the held-out gear, which is then scored.

    Also returns each fold's alarm threshold per HI (index = held-out gear): the alarm_q quantile of the HI on the
    fit gears' own baseline minutes.
    """
    parts, thr = [], {}
    for held in SPLITS["train"]:
        fit = z_tr[z_tr.experiment != held]
        hm = HealthyModel().fit(fit)
        parts.append(hm.score(z_tr[z_tr.experiment == held]))
        thr[held] = hm.score(fit[fit.base]).drop(columns=["experiment", "run_num", "hours"]).quantile(alarm_q)
    return pd.concat(parts), pd.DataFrame(thr).T


def label_free_table(s: pd.DataFrame, thr: pd.DataFrame, ends: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Score candidate HIs without damage labels (PHM health-index criteria, after Coble & Hines).

    Uses two facts from the test design, not labels: every training gear was run until it spalled, and its first
    run is healthy (the baseline window sits inside it). Computed on raw run-end values (monotone="none"), since
    monotone smoothing would make every HI trivially monotone.
      monotonicity    mean over gears of |sum of run-to-run changes| / sum of |changes|: 1 if the HI never
                      falls. Weighted by size, unlike Coble & Hines' |#rises - #falls| / (n - 1), so an HI that
                      stays flat on healthy runs and jumps at spall is not penalised for flat stretches or jitter.
      trendability    smallest Pearson correlation between two gears' trajectories on a common 0..1 life axis
      prognosability  exp(-std(final values) / mean |final - first|): do gears end at a similar level? This is
                      the label-free stand-in for the pooled metric's need for one scale across gears.
      false_alarm     mean over gears of the share of run-1 minutes above the fold's alarm threshold
      fitness         (monotonicity + max(trendability, 0) + prognosability) / 3 - false_alarm
    """
    r = run_end_scores(s, ends, cols, "none")
    life = np.linspace(0, 1, 21)
    run1 = s[s.run_num == 1]
    rows = []
    for c in cols:
        mono, traj, first, last = [], [], [], []
        for _, g in r.groupby("experiment"):
            v = g[c].to_numpy()
            d = np.diff(v)
            mono.append(abs(d.sum()) / max(np.abs(d).sum(), 1e-9))
            traj.append(np.interp(life, g.end_h / g.end_h.max(), v))
            first.append(v[0])
            last.append(v[-1])
        corr = np.nan_to_num(np.corrcoef(traj), nan=0.0)  # a flat trajectory has no trend
        rise = max(np.mean(np.abs(np.subtract(last, first))), 1e-9)
        alarm = run1[c] > run1.experiment.map(thr[c])
        rows.append(
            {
                "hi": c,
                "monotonicity": float(np.mean(mono)),
                "trendability": float(corr[np.triu_indices(len(traj), 1)].min()),
                "prognosability": float(np.exp(-np.std(last) / rise)),
                "false_alarm": float(alarm.groupby(run1.experiment).mean().mean()),
            }
        )
    t = pd.DataFrame(rows).set_index("hi")
    t["fitness"] = (t.monotonicity + t.trendability.clip(lower=0) + t.prognosability) / 3 - t.false_alarm
    return t.round(4)


def evaluate(z_tr, ends_tr, debris_tr, labels: Path, cols, monotone="iso") -> pd.DataFrame:
    """Leave-one-gear-out scores of each candidate against the condition labels (evaluation only)."""
    s, _ = logo_scores(z_tr)
    r = run_end_scores(s, ends_tr, cols, monotone).merge(debris_tr, on=["experiment", "run_num"], how="left")
    # debris relative to each gear's run-1 rate, made monotone like the vibration HIs
    r["debris"] = r.debris_rate / r.groupby("experiment").debris_rate.transform("first")
    r["debris"] = r.groupby("experiment").debris.ffill()
    r["debris"] = r.groupby("experiment").debris.cummax()
    truth = label_damage(r, labels)
    rows = []
    for c in cols + ["debris", "hours_proxy"]:
        pred = r.end_h.to_numpy() if c == "hours_proxy" else r[c].to_numpy()
        rows.append({"hi": c, **challenge_score(pred, truth, r.experiment.to_numpy(), MAX_DAMAGE)})
    return pd.DataFrame(rows).set_index("hi").round(4), r.assign(damage=truth)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("models/out"))
    ap.add_argument("--monotone", default="iso", choices=["none", "cummax", "iso"])
    ap.add_argument("--hi", default=HI_DEFAULT, help="HI column to submit with --select fixed")
    ap.add_argument(
        "--select",
        default="fixed",
        choices=["fixed", "label_free", "labels"],
        help="fixed: submit --hi; label_free: best label-free fitness; labels: lowest LOGO pooled MSE vs the condition "
        "labels (uses labels for model selection)",
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["maha", "spe", "zpos", "zmax", "tooth", "tooth_maha"]

    z = {s: condition_correct(load_minutes(a.data, s)) for s in SPLITS}
    ends = {s: run_ends(a.data, s) for s in SPLITS}

    debris = load_debris(a.data, "train")
    labels = a.data / "labels" / "condition_labels.csv"
    for mode in ["none", "cummax", "iso"]:
        t, _ = evaluate(z["train"], ends["train"], debris, labels, cols, mode)
        print(f"\n[{mode}] leave-one-gear-out vs condition labels (lower mse / higher spearman is better):")
        print(t.to_string())
    table, r_tr = evaluate(z["train"], ends["train"], debris, labels, cols, a.monotone)
    print(r_tr[["experiment", "run_num", "end_h", "damage"] + cols].round(2).to_string(index=False))
    s_tr, thr = logo_scores(z["train"])
    lf = label_free_table(s_tr, thr, ends["train"], cols)
    print("\nlabel-free selection criteria (no labels; higher fitness is better):")
    print(lf.to_string())
    best = {
        "fixed": a.hi,
        "label_free": lf.fitness.idxmax(),
        "labels": table.loc[cols, "mse_pooled"].idxmin(),
    }[a.select]
    print(f"\nsubmitting HI = {best}  (--select {a.select})")

    hm = HealthyModel().fit(z["train"])
    for split, tmpl in [("validation", "example_validation_submission.csv"), ("test", "example_submission.csv")]:
        r = run_end_scores(hm.score(z[split]), ends[split], cols, a.monotone)
        sub = pd.read_csv(a.data / split / tmpl)
        r["run_id"] = r.experiment.str[-1]
        # match each template time to the nearest run end of that gear
        sub["metric"] = [
            r.loc[(r.run_id == g), [best, "end_h"]].pipe(lambda x: x[best].iloc[(x.end_h - t).abs().argmin()])
            for g, t in zip(sub.run_id, sub.time)
        ]
        sub["metric"] = sub.metric.round(4)
        sub.to_csv(a.out / f"{split}_submission.csv", index=False)
        print(f"\n{split}:\n{sub.to_string(index=False)}")
    table.to_csv(a.out / "logo_eval.csv")
    lf.to_csv(a.out / "label_free_eval.csv")
    json.dump(
        {
            "hi": best,
            "select": a.select,
            "monotone": a.monotone,
            "feats": FEATS,
            "tooth": TOOTH,
            "base_h": [BREAK_IN_H, BASE_H],
            "tail_min": TAIL_MIN,
            "tail_q": TAIL_Q,
        },
        open(a.out / "config.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
