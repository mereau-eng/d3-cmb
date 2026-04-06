#!/usr/bin/env python3
"""
Cross-multipole phase locking null test.

Distinguishes "boundary filter" (IPT) from "rigid stencil" (artifact):
  - Boundary filter: D3 constrains power fractions, not phases. Phases are random.
  - Rigid stencil: a fixed template imposes phase correlations across l at the same m.

For each m with m mod 3 == 0 (A2-relevant modes), we collect the complex phases
arg(a_{l,m}) across all l values in the D3 frame and test for uniformity using
the Rayleigh test. Results are combined across m via Fisher's method.

E-mode channels (m mod 3 != 0) serve as a control: these should show no phase
locking under either hypothesis.

Usage:
  python d3_phase_locking_test.py --fits map1.fits [map2.fits ...] --lmax 150
"""

import numpy as np
import healpy as hp
import json, time, os, argparse
from scipy.stats import chi2

D3_ANTI_L = 230.3
D3_ANTI_B = 64.9


def rotate_to_d3_frame(alm, lmax):
    theta_d3 = np.radians(90.0 - D3_ANTI_B)
    phi_d3 = np.radians(D3_ANTI_L)
    alm_d3 = alm.copy()
    hp.rotate_alm(alm_d3, -phi_d3, -theta_d3, 0, lmax=lmax)
    return alm_d3


def rayleigh_test(phases):
    """Rayleigh test for uniformity of circular data.
    Returns mean resultant length R and p-value."""
    n = len(phases)
    if n < 3:
        return n, np.nan, 1.0
    R = np.abs(np.mean(np.exp(1j * phases)))
    p = np.exp(-n * R**2)  # exact for large n
    return n, float(R), float(p)


def pairwise_phase_coherence(phases):
    """Mean cosine of pairwise phase differences.
    Under uniformity, E[cos(phi_i - phi_j)] = 0."""
    n = len(phases)
    if n < 3:
        return np.nan
    e = np.exp(1j * phases)
    # R^2 = |mean(e)|^2 = mean(cos) of pairwise diffs (up to n normalization)
    R = np.abs(np.mean(e))
    return float(R**2)


def run_analysis(fits_paths, lmax=150, n_sims=10000, outdir='extended_results'):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)

    m_a2 = list(range(3, min(51, lmax + 1), 3))      # m = 3,6,...,48
    m_e = [m for m in range(1, min(51, lmax + 1)) if m % 3 != 0]  # control

    all_results = {}

    for fpath in fits_paths:
        label = None
        fname = os.path.basename(fpath).lower()
        for name in ['smica', 'nilc', 'sevem', 'commander']:
            if name in fname:
                label = name.upper()
                break
        if label is None:
            label = os.path.splitext(os.path.basename(fpath))[0].upper()

        print(f"\n{'='*72}")
        print(f"PHASE LOCKING TEST: {label}")
        print(f"  lmax={lmax}, N_MC={n_sims}")
        print(f"  A2 modes tested: m = {m_a2[0]}, {m_a2[1]}, ..., {m_a2[-1]}")
        print(f"  E control modes: {len(m_e)} values")
        print(f"{'='*72}")

        # Load map and rotate to D3 frame
        print(f"\n[{time.time()-t0:.1f}s] Loading and rotating...")
        sky_map = hp.read_map(fpath, field=0)
        if hp.get_nside(sky_map) > 256 and lmax <= 384:
            sky_map = hp.ud_grade(sky_map, 256)
        alm_gal = hp.map2alm(sky_map, lmax=lmax)
        alm_d3 = rotate_to_d3_frame(alm_gal, lmax)

        # ---- A2-RELEVANT MODES (m mod 3 == 0) ----
        print(f"\n  A2-relevant modes (m mod 3 == 0, m > 0):")
        print(f"  {'m':>4} {'n_l':>6} {'R':>8} {'p':>10}")
        print(f"  {'-'*32}")

        a2_per_m = {}
        a2_pvals = []
        for m in m_a2:
            phases = []
            for l in range(max(m, 2), lmax + 1):
                idx = hp.Alm.getidx(lmax, l, m)
                c = alm_d3[idx]
                if abs(c) > 1e-30:
                    phases.append(np.angle(c))
            phases = np.array(phases)

            n, R, p = rayleigh_test(phases)
            a2_per_m[m] = {'n': n, 'R': R, 'p': p}
            if not np.isnan(R):
                a2_pvals.append(p)

            flag = " *" if p < 0.05 else (" **" if p < 0.01 else "")
            print(f"  {m:>4} {n:>6} {R:>8.4f} {p:>10.4f}{flag}")

        # Fisher combined for A2 modes
        a2_fisher = -2 * sum(np.log(max(p, 1e-15)) for p in a2_pvals)
        a2_dof = 2 * len(a2_pvals)
        a2_fisher_p = chi2.sf(a2_fisher, a2_dof)
        n_sig_a2 = sum(1 for p in a2_pvals if p < 0.05)

        print(f"\n  Fisher combined (A2): stat={a2_fisher:.2f}, dof={a2_dof}, "
              f"p={a2_fisher_p:.4f}")
        print(f"  Significant at 5%: {n_sig_a2}/{len(a2_pvals)} "
              f"(expected by chance: {len(a2_pvals)*0.05:.1f})")

        # ---- E-MODE CONTROL (m mod 3 != 0) ----
        e_pvals = []
        n_sig_e = 0
        for m in m_e:
            phases = []
            for l in range(max(m, 2), lmax + 1):
                idx = hp.Alm.getidx(lmax, l, m)
                c = alm_d3[idx]
                if abs(c) > 1e-30:
                    phases.append(np.angle(c))
            phases = np.array(phases)
            _, _, p = rayleigh_test(phases)
            if not np.isnan(p):
                e_pvals.append(p)
                if p < 0.05:
                    n_sig_e += 1

        e_fisher = -2 * sum(np.log(max(p, 1e-15)) for p in e_pvals)
        e_dof = 2 * len(e_pvals)
        e_fisher_p = chi2.sf(e_fisher, e_dof)

        print(f"\n  E-mode control (m mod 3 != 0):")
        print(f"  Fisher combined: stat={e_fisher:.2f}, dof={e_dof}, p={e_fisher_p:.4f}")
        print(f"  Significant at 5%: {n_sig_e}/{len(e_pvals)} "
              f"(expected: {len(e_pvals)*0.05:.1f})")

        # ---- MC NULL FOR A2 FISHER STATISTIC ----
        print(f"\n[{time.time()-t0:.1f}s] MC null distribution (N={n_sims})...")
        rng = np.random.default_rng(42)
        mc_fisher = np.zeros(n_sims)

        for i in range(n_sims):
            mc_pvals_i = []
            for m in m_a2:
                n_l = lmax - max(m, 2) + 1
                if n_l < 3:
                    continue
                # Random Gaussian alms -> uniform phases
                re = rng.standard_normal(n_l)
                im = rng.standard_normal(n_l)
                ph = np.arctan2(im, re)
                _, _, p = rayleigh_test(ph)
                if not np.isnan(p):
                    mc_pvals_i.append(p)
            mc_fisher[i] = -2 * sum(np.log(max(p, 1e-15)) for p in mc_pvals_i)

        mc_pte = float(np.mean(mc_fisher >= a2_fisher))
        mc_mean = float(np.mean(mc_fisher))
        mc_std = float(np.std(mc_fisher))
        mc_z = (a2_fisher - mc_mean) / mc_std if mc_std > 0 else 0

        print(f"  Observed Fisher stat: {a2_fisher:.2f}")
        print(f"  MC mean: {mc_mean:.2f}, MC std: {mc_std:.2f}")
        print(f"  MC PTE: {mc_pte:.4f} (z = {mc_z:+.2f})")

        verdict = "NULL CONFIRMED (no phase locking)" if mc_pte > 0.05 else "PHASE LOCKING DETECTED"
        print(f"\n  VERDICT: {verdict}")

        # ---- LOW-l vs HIGH-l SPLIT ----
        print(f"\n  Phase coherence by l-range (per m, Rayleigh R):")
        print(f"  {'m':>4} {'R(l<=30)':>10} {'R(l>30)':>10} {'R(all)':>10}")
        print(f"  {'-'*38}")
        for m in m_a2[:8]:  # show first 8
            ph_low, ph_high = [], []
            for l in range(max(m, 2), lmax + 1):
                idx = hp.Alm.getidx(lmax, l, m)
                c = alm_d3[idx]
                if abs(c) > 1e-30:
                    if l <= 30:
                        ph_low.append(np.angle(c))
                    else:
                        ph_high.append(np.angle(c))
            _, R_low, _ = rayleigh_test(np.array(ph_low)) if len(ph_low) >= 3 else (0, np.nan, 1)
            _, R_high, _ = rayleigh_test(np.array(ph_high)) if len(ph_high) >= 3 else (0, np.nan, 1)
            R_all = a2_per_m[m]['R']
            print(f"  {m:>4} {R_low:>10.4f} {R_high:>10.4f} {R_all:>10.4f}")

        all_results[label] = {
            'a2_fisher': float(a2_fisher),
            'a2_fisher_p': float(a2_fisher_p),
            'a2_fisher_dof': a2_dof,
            'a2_mc_pte': mc_pte,
            'a2_mc_z': float(mc_z),
            'n_sig_a2_005': n_sig_a2,
            'n_a2_modes': len(a2_pvals),
            'e_fisher': float(e_fisher),
            'e_fisher_p': float(e_fisher_p),
            'n_sig_e_005': n_sig_e,
            'n_e_modes': len(e_pvals),
            'per_m_a2': {str(m): a2_per_m[m] for m in m_a2},
        }

    # ---- CROSS-MAP SUMMARY ----
    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print("CROSS-MAP SUMMARY")
        print(f"{'='*72}")
        print(f"  {'Map':>12} {'A2 Fisher':>12} {'A2 p':>10} {'MC PTE':>10} {'E Fisher p':>12} {'Verdict':>20}")
        print(f"  {'-'*78}")
        for lab, r in all_results.items():
            v = "null" if r['a2_mc_pte'] > 0.05 else "LOCKED"
            print(f"  {lab:>12} {r['a2_fisher']:>12.2f} {r['a2_fisher_p']:>10.4f} "
                  f"{r['a2_mc_pte']:>10.4f} {r['e_fisher_p']:>12.4f} {v:>20}")

    outfile = os.path.join(outdir, 'phase_locking_test.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")
    print(f"Total time: {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='D3 Phase Locking Null Test')
    parser.add_argument('--fits', nargs='+', required=True, help='FITS map paths')
    parser.add_argument('--lmax', type=int, default=150)
    parser.add_argument('--nsims', type=int, default=10000)
    parser.add_argument('--outdir', default='extended_results')
    args = parser.parse_args()
    run_analysis(args.fits, args.lmax, args.nsims, args.outdir)


if __name__ == '__main__':
    main()
