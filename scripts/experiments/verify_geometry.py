"""Recompute every geometry number quoted in the findings note, from the traces."""
import os

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/scratch/vocabulary-growth/output/models"
DIRS = {
    "VG10": "VG10-age-understood-spoken-ds-re-subj-uq-anchored",
    "VG12": "VG12-age-understood-td-re",
    "VG13": "VG13-age-understood-spoken-td-re-young",
}


def load(model):
    return az.from_netcdf(os.path.join(ROOT, DIRS[model], "trace.nc"))


def flat(idata, name):
    return idata.posterior[name].values.reshape(-1)


def energy(idata):
    return idata.sample_stats["energy"].values.reshape(-1)


def scalar_names(idata):
    return [
        n for n, v in idata.posterior.data_vars.items()
        if v.values.ndim == 2  # (chain, draw) only -> scalar parameter
    ]


print("=" * 70)
print("1. VG12 energy correlations (top 12 scalar parameters)")
print("=" * 70)
i12 = load("VG12")
e = energy(i12)
rows = []
for n in scalar_names(i12):
    x = flat(i12, n)
    if np.std(x) == 0:
        continue
    rows.append((n, float(np.corrcoef(x, e)[0, 1])))
rows.sort(key=lambda r: -abs(r[1]))
for n, c in rows[:12]:
    print(f"  {n:26s} {c:+.3f}")
print("  ... 'tau' present?", [f"{c:+.3f}" for n, c in rows if n == "tau"])
print("  rank of 'tau':", [i for i, (n, _) in enumerate(rows, 1) if n == "tau"])

print()
print("=" * 70)
print("2. Worst ESS in VG12")
print("=" * 70)
ess = az.ess(i12, var_names=scalar_names(i12))
vals = sorted(((n, float(ess[n].values)) for n in ess.data_vars), key=lambda r: r[1])
for n, v in vals[:6]:
    print(f"  {n:26s} {v:8.0f}")

print()
print("=" * 70)
print("3. Singleton falsification: VG10 (DS) vs VG12 (TD)")
print("=" * 70)
i10 = load("VG10")
for model, idata, tau_s, kap in [
    ("VG10", i10, "tau_subj_u", "kappa_young_u"),
    ("VG12", i12, "tau_subject", "kappa_young"),
]:
    have = set(idata.posterior.data_vars)
    if tau_s not in have or kap not in have:
        print(f"  {model}: MISSING {tau_s if tau_s not in have else kap}")
        print(f"    available tau_*: {[n for n in have if n.startswith('tau')]}")
        print(f"    available kappa_*: {[n for n in have if n.startswith('kappa') and idata.posterior[n].values.ndim == 2]}")
        continue
    a, b = flat(idata, tau_s), flat(idata, kap)
    ee = energy(idata)
    # BFMI per chain, computed directly: sum of squared successive energy
    # differences over the energy variance (Betancourt 2016).
    e2d = idata.sample_stats["energy"].values  # (chain, draw)
    bfmi = float(
        np.min(
            [
                np.sum(np.diff(row) ** 2) / np.sum((row - row.mean()) ** 2)
                for row in e2d
            ]
        )
    )
    print(f"  {model}: corr({tau_s}, {kap}) = {np.corrcoef(a, b)[0,1]:+.3f}   "
          f"corr({tau_s}, energy) = {np.corrcoef(a, ee)[0,1]:+.3f}   min BFMI = {bfmi:.3f}")

print()
print("=" * 70)
print("4. HSGP lengthscale health (VG12)")
print("=" * 70)
for n in [v for v in i12.posterior.data_vars if v.startswith("ell")]:
    x = flat(i12, n)
    lo, hi = np.quantile(x, [0.055, 0.945])  # 89% equal-tailed interval
    print(f"  {n:16s} mean {x.mean():.3f}  89% ETI [{lo:.3f}, {hi:.3f}]")

print()
print("=" * 70)
print("5. eta prior-data conflict (contraction = 1 - post_sd / prior_sd)")
print("=" * 70)
HN_SD = np.sqrt(1 - 2 / np.pi)
TARGETS = [
    ("VG12", i12, "eta", 0.5),
    ("VG13", load("VG13"), "eta_u", 0.4),
    ("VG13", None, "eta_q", 0.2),
    ("VG10", i10, "eta_u", 0.6),
    ("VG10", None, "eta_q", 0.8),
]
cache = {}
print(f"  {'model':6s} {'param':8s} {'sigma':>6s} {'post mean':>10s} {'post sd':>8s} {'priorCDF':>9s} {'contract':>9s}")
last = None
for model, idata, param, sigma in TARGETS:
    if idata is not None:
        cache[model] = idata
    idata = cache[model]
    if param not in idata.posterior.data_vars:
        print(f"  {model:6s} {param:8s} -- absent --")
        continue
    x = flat(idata, param)
    cdf = float(stats.halfnorm.cdf(x.mean(), scale=sigma))
    contraction = 1 - x.std() / (sigma * HN_SD)
    print(f"  {model:6s} {param:8s} {sigma:6.2f} {x.mean():10.3f} {x.std():8.3f} {cdf:9.3f} {contraction:9.3f}")

print()
print("=" * 70)
print("6. Repeat-measurement rates")
print("=" * 70)
for model in ["VG10", "VG12"]:
    p = os.path.join(ROOT, DIRS[model], "analysis_data.csv")
    if not os.path.isfile(p):
        # fall back to whatever frame the fit wrote
        cands = [f for f in os.listdir(os.path.join(ROOT, DIRS[model])) if f.endswith(".csv")]
        print(f"  {model}: no analysis_data.csv; csvs = {cands[:10]}")
        continue
    df = pd.read_csv(p)
    key = "subject_code" if "subject_code" in df else "subject_key"
    sizes = df.groupby(key).size()
    print(f"  {model}: {int((sizes > 1).sum())} repeat / {len(sizes)} subjects "
          f"({(sizes > 1).mean():.1%}), {len(df)} rows")
