"""Choose variance-partition priors whose induced marginals match VG12's current ones.

Current VG12 priors:
    tau_subject        ~ HalfNormal(1.5)
    kappa_excess_young ~ LogNormal(log 40, 0.9)     (young anchor at 12 months)

Target parameterisation:
    v_total ~ LogNormal(total_mu, total_sigma)
    share   ~ Beta(share_alpha, share_beta)
    tau_subject        = sqrt(share * v_total)
    kappa_excess_young = c / ((1 - share) * v_total),  c = 1/(p0 (1-p0))

We do not need an exact match -- the prior is *meant* to move onto the budget and
the split -- but the induced marginals should be recognisably the same beliefs,
otherwise the reparameterisation smuggles in a different model.
"""
import numpy as np
from scipy import optimize, stats

RNG = np.random.default_rng(20260805)
N = 200_000

# Observed proportion at VG12's young kappa anchor (12 months), from the data.
P0 = None  # filled in by the caller below


def induced(total_mu, total_sigma, a, b, p0, n=N):
    c = 1.0 / (p0 * (1 - p0))
    v = RNG.lognormal(mean=total_mu, sigma=total_sigma, size=n)
    s = RNG.beta(a, b, size=n)
    tau = np.sqrt(s * v)
    excess = c / ((1 - s) * v)
    return tau, excess


def summarise(name, x, ref_q):
    q = np.quantile(x, [0.05, 0.5, 0.95])
    print(f"  {name:20s} 5%/50%/95% = {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f}"
          f"   (target {ref_q[0]:.3f} {ref_q[1]:.3f} {ref_q[2]:.3f})")
    return q


def main(p0):
    # Target quantiles from the current priors.
    tau_ref = stats.halfnorm.ppf([0.05, 0.5, 0.95], scale=1.5)
    exc_ref = stats.lognorm.ppf([0.05, 0.5, 0.95], s=0.9, scale=np.exp(np.log(40.0)))
    print(f"reference proportion p0 = {p0:.4f}  ->  c = {1/(p0*(1-p0)):.3f}")
    print("targets:")
    print(f"  tau_subject        5/50/95 = {tau_ref[0]:.3f} {tau_ref[1]:.3f} {tau_ref[2]:.3f}")
    print(f"  kappa_excess_young 5/50/95 = {exc_ref[0]:.3f} {exc_ref[1]:.3f} {exc_ref[2]:.3f}")
    print()

    def loss(theta):
        total_mu, log_total_sigma, log_a, log_b = theta
        tau, exc = induced(total_mu, np.exp(log_total_sigma),
                           np.exp(log_a), np.exp(log_b), p0, n=40_000)
        tq = np.quantile(tau, [0.05, 0.5, 0.95])
        eq = np.quantile(exc, [0.05, 0.5, 0.95])
        # Match on the log scale: both quantities are positive and span decades.
        return float(
            np.sum((np.log(tq) - np.log(tau_ref)) ** 2)
            + np.sum((np.log(eq) - np.log(exc_ref)) ** 2)
        )

    best = None
    for seed_theta in [
        (np.log(1.3), np.log(0.8), np.log(3.0), np.log(1.0)),
        (np.log(2.0), np.log(1.0), np.log(2.0), np.log(2.0)),
        (np.log(1.0), np.log(0.6), np.log(5.0), np.log(1.5)),
    ]:
        res = optimize.minimize(loss, seed_theta, method="Nelder-Mead",
                                options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res

    total_mu, log_total_sigma, log_a, log_b = best.x
    total_sigma = np.exp(log_total_sigma)
    a, b = np.exp(log_a), np.exp(log_b)
    print(f"fitted: total_mu={total_mu:.4f} total_sigma={total_sigma:.4f} "
          f"share_alpha={a:.4f} share_beta={b:.4f}   loss={best.fun:.5f}")
    print()
    tau, exc = induced(total_mu, total_sigma, a, b, p0)
    print("induced marginals:")
    summarise("tau_subject", tau, tau_ref)
    summarise("kappa_excess_young", exc, exc_ref)
    print()
    print(f"  implied share    5/50/95 = "
          f"{np.quantile(RNG.beta(a,b,size=N), [0.05,0.5,0.95])}")
    print(f"  implied v_total  5/50/95 = "
          f"{np.quantile(RNG.lognormal(total_mu, total_sigma, size=N), [0.05,0.5,0.95])}")
    print()
    print("ROUNDED FOR THE DEFINITION:")
    print(f"    reference_proportion={p0:.4f},")
    print(f"    total_mu={round(float(total_mu), 2)},")
    print(f"    total_sigma={round(float(total_sigma), 2)},")
    print(f"    share_alpha={round(float(a), 2)},")
    print(f"    share_beta={round(float(b), 2)},")
    # Report the rounded version's induced marginals -- that is what ships.
    tau, exc = induced(round(float(total_mu), 2), round(float(total_sigma), 2),
                       round(float(a), 2), round(float(b), 2), p0)
    print()
    print("induced marginals AT THE ROUNDED VALUES (what ships):")
    summarise("tau_subject", tau, tau_ref)
    summarise("kappa_excess_young", exc, exc_ref)


if __name__ == "__main__":
    import sys
    main(float(sys.argv[1]))
