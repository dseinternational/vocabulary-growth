"""Compare the VG12 geometry arms on the diagnostics that motivated the changes."""
import json
import os

import arviz as az
import numpy as np

ROOT = "/scratch/vg-geom-output/models"
ARMS = ["baseline", "eta", "centred", "partition"]


def bfmi_per_chain(idata):
    e = idata.sample_stats["energy"].values
    return np.array(
        [np.sum(np.diff(r) ** 2) / np.sum((r - r.mean()) ** 2) for r in e]
    )


def energy_corr(idata, name):
    if name not in idata.posterior.data_vars:
        return np.nan
    x = idata.posterior[name].values.reshape(-1)
    e = idata.sample_stats["energy"].values.reshape(-1)
    if np.std(x) == 0:
        return np.nan
    return float(np.corrcoef(x, e)[0, 1])


def scalar_names(idata):
    return [
        n for n, v in idata.posterior.data_vars.items() if v.values.ndim == 2
    ]


rows = []
for arm in ARMS:
    d = os.path.join(ROOT, f"VG12-geom-{arm}")
    p = os.path.join(d, "trace.nc")
    if not os.path.isfile(p):
        print(f"{arm:10s} NO TRACE at {p}")
        continue
    idata = az.from_netcdf(p)
    b = bfmi_per_chain(idata)
    div = int(idata.sample_stats["diverging"].values.sum())

    names = scalar_names(idata)
    ess = az.ess(idata, var_names=names)
    ess_vals = {n: float(ess[n].values) for n in ess.data_vars}
    worst = sorted(ess_vals.items(), key=lambda kv: kv[1])[:3]

    rhat = az.rhat(idata, var_names=names)
    max_rhat = max(float(rhat[n].values) for n in rhat.data_vars)

    # The ridge itself, where both parameters still exist by name.
    post = idata.posterior
    if "tau_subject" in post and "kappa_young" in post:
        a = post["tau_subject"].values.reshape(-1)
        c = post["kappa_young"].values.reshape(-1)
        ridge = float(np.corrcoef(a, c)[0, 1])
    else:
        ridge = np.nan

    rows.append(
        dict(
            arm=arm,
            min_bfmi=float(b.min()),
            mean_bfmi=float(b.mean()),
            divergences=div,
            max_rhat=max_rhat,
            ess_tau=ess_vals.get("tau", np.nan),
            ess_min=worst[0][1],
            ess_min_param=worst[0][0],
            ridge=ridge,
            e_tau_subject=energy_corr(idata, "tau_subject"),
            e_kappa_young=energy_corr(idata, "kappa_young"),
            e_v_total=energy_corr(idata, "v_total"),
            e_share=energy_corr(idata, "subject_variance_share"),
        )
    )

print()
print("=" * 108)
print(f"{'arm':10s} {'minBFMI':>8s} {'meanBFMI':>9s} {'div':>5s} {'maxRhat':>8s} "
      f"{'ESS tau':>8s} {'ESS min':>8s} {'(param)':>20s} {'ridge':>7s}")
print("=" * 108)
for r in rows:
    print(f"{r['arm']:10s} {r['min_bfmi']:8.3f} {r['mean_bfmi']:9.3f} "
          f"{r['divergences']:5d} {r['max_rhat']:8.4f} {r['ess_tau']:8.0f} "
          f"{r['ess_min']:8.0f} {r['ess_min_param']:>20s} {r['ridge']:7.3f}")

print()
print("energy correlations (the BFMI drivers):")
print(f"{'arm':10s} {'tau_subject':>12s} {'kappa_young':>12s} {'v_total':>9s} {'share':>9s}")
for r in rows:
    def f(v):
        return "     --  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+9.3f}"
    print(f"{r['arm']:10s} {f(r['e_tau_subject']):>12s} {f(r['e_kappa_young']):>12s} "
          f"{f(r['e_v_total']):>9s} {f(r['e_share']):>9s}")

# Estimands must survive the reparameterisation: tau_subject is a reported
# cross-population quantity, so check the arms agree on it.
print()
print("tau_subject posterior (the reported heterogeneity estimand):")
for arm in ARMS:
    p = os.path.join(ROOT, f"VG12-geom-{arm}", "trace.nc")
    if not os.path.isfile(p):
        continue
    x = az.from_netcdf(p).posterior["tau_subject"].values.reshape(-1)
    q = np.quantile(x, [0.055, 0.5, 0.945])
    print(f"  {arm:10s} mean {x.mean():.4f} sd {x.std():.4f}  89% ETI [{q[0]:.4f}, {q[2]:.4f}]")

with open(os.path.join(os.path.dirname(__file__), "geom", "summary.json"), "w") as fh:
    json.dump(rows, fh, indent=2)
