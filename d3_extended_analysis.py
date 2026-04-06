#!/usr/bin/env python3
"""
Extended D3 CMB Validation
==========================
Addresses team priorities:
  (2) Extend to lmax=150 (NSIDE=256) — test signal persistence to l~150
  (1) High-resolution MC (N=10,000) — tighten PTE estimates
  (4) Cross-map validation — run on SMICA, NILC, SEVEM, Commander

Usage:
  python d3_extended_analysis.py --fits <path> --lmax 150 --nsims 10000 --outdir results/
  python d3_extended_analysis.py --fits <path> --lmax 150 --nsims 10000 --gamma-opt --outdir results/

Author: Robert Mereau / Claude analysis extension
Date: April 2026
"""

import numpy as np
import healpy as hp
import json
import argparse
import time
import os
from scipy.stats import chi2


# =====================================================================
# D3 AXIS (from prior NSIDE=32 optimization)
# =====================================================================
D3_AXIS_L = 50.3       # Galactic longitude (degrees)
D3_AXIS_B = -64.9      # Galactic latitude (degrees)
D3_ANTI_L = 230.3      # Antipode longitude
D3_ANTI_B = 64.9       # Antipode latitude


# =====================================================================
# ROTATION: Galactic -> D3 frame
# =====================================================================
def rotate_to_d3_frame(alm, lmax):
    """Rotate alm from Galactic coordinates to D3 frame (D3 axis = Z).

    Uses the antipode convention: brings (230.3, +64.9) to the north pole.
    Verified to give f_{m=+-3}(l=3) = 0.94 matching prior analysis.
    """
    theta_d3 = np.radians(90.0 - D3_ANTI_B)   # colatitude = 25.1 deg
    phi_d3 = np.radians(D3_ANTI_L)             # longitude = 230.3 deg
    alm_d3 = alm.copy()
    hp.rotate_alm(alm_d3, -phi_d3, -theta_d3, 0, lmax=lmax)
    return alm_d3


# =====================================================================
# D3 IRREP DECOMPOSITION (efficient m-basis method)
# =====================================================================
def d3_irrep_dimensions(l):
    """Theoretical D3 irrep subspace dimensions for multipole l.

    Returns (dim_A1, dim_A2, dim_E) and expected fractions.
    """
    dim_A1 = 1  # m=0 always contributes 1 to A1
    dim_A2 = 0
    for m in range(1, l + 1):
        if m % 3 == 0:
            dim_A1 += 1  # Re part of +-m pair
            dim_A2 += 1  # Im part of +-m pair
        # else: E gets 2 (the +-m pair)
    dim_E = (2 * l + 1) - dim_A1 - dim_A2
    total = 2 * l + 1
    return dim_A1, dim_A2, dim_E, dim_A1/total, dim_A2/total, dim_E/total


def d3_fractions(alm_d3, lmax_alm, l, gamma=0.0):
    """Compute D3 irrep power fractions from alm in D3 frame.

    In the D3 frame (D3 axis = Z-axis), after azimuthal phase gamma:
      E irrep:  all m with m mod 3 != 0  (independent of gamma)
      A1 irrep: Re^2 part of m mod 3 == 0 pairs + m=0
      A2 irrep: Im^2 part of m mod 3 == 0 pairs

    Args:
        alm_d3: healpy alm array in D3 frame
        lmax_alm: maximum l used in alm extraction
        l: multipole to analyze
        gamma: azimuthal phase for reflection planes (radians)

    Returns:
        (f_A1, f_A2, f_E) power fractions (sum to 1)
    """
    total = A1 = A2 = E = 0.0
    for m in range(0, l + 1):
        idx = hp.Alm.getidx(lmax_alm, l, m)
        c = alm_d3[idx]
        if gamma != 0.0 and m > 0:
            c = c * np.exp(-1j * m * gamma)

        if m == 0:
            p = np.abs(c)**2
            total += p
            A1 += p  # m=0 is always A1
        else:
            p = 2.0 * np.abs(c)**2  # factor 2 for +-m pair
            total += p
            if m % 3 == 0:
                # A1/A2 split by Re/Im of the rotated coefficient
                A1 += 2.0 * np.real(c)**2
                A2 += 2.0 * np.imag(c)**2
            else:
                E += p

    if total == 0:
        return 0.0, 0.0, 0.0
    return A1 / total, A2 / total, E / total


def d3_fractions_from_raw(alm_vec_l, l, gamma=0.0):
    """D3 fractions from a raw complex vector of CS-constrained alms.

    alm_vec_l: array of length l+1, containing a_{l,m} for m=0..l
               with CS convention a_{l,-m} = (-1)^m conj(a_{l,m})
    """
    total = A1 = A2 = E = 0.0
    for m in range(0, l + 1):
        c = alm_vec_l[m]
        if gamma != 0.0 and m > 0:
            c = c * np.exp(-1j * m * gamma)

        if m == 0:
            p = np.abs(c)**2
            total += p
            A1 += p
        else:
            p = 2.0 * np.abs(c)**2
            total += p
            if m % 3 == 0:
                A1 += 2.0 * np.real(c)**2
                A2 += 2.0 * np.imag(c)**2
            else:
                E += p

    if total == 0:
        return 0.0, 0.0, 0.0
    return A1 / total, A2 / total, E / total


# =====================================================================
# GAMMA OPTIMIZATION
# =====================================================================
def optimize_gamma(alm_d3, lmax_alm, l_range, n_steps=360):
    """Find azimuthal phase gamma that maximizes mean f_A2.

    Scans gamma in [0, 2pi/3) with n_steps resolution.
    D3 has 3-fold symmetry so gamma is periodic with period 2pi/3.
    """
    best_gamma = 0.0
    best_score = -1.0

    for i in range(n_steps):
        gamma = (2 * np.pi / 3) * i / n_steps
        score = 0.0
        for l in l_range:
            _, f2, _ = d3_fractions(alm_d3, lmax_alm, l, gamma)
            _, _, _, _, ef2, _ = d3_irrep_dimensions(l)
            score += (f2 - ef2)  # sum of A2 excess
        if score > best_score:
            best_score = score
            best_gamma = gamma

    return best_gamma, best_score


# =====================================================================
# MONTE CARLO (no rotation needed for isotropic null)
# =====================================================================
def mc_d3_fractions(l, n_sims, gamma=0.0, rng=None):
    """Generate MC distribution of D3 fractions for multipole l.

    Key insight: for isotropic Gaussian random fields, the D3 fraction
    distribution is the same regardless of frame (by rotational invariance).
    So we generate random alms directly — no rotation needed.

    Returns arrays (fA1_mc, fA2_mc, fE_mc) each of length n_sims.
    """
    if rng is None:
        rng = np.random.default_rng()

    fA1 = np.empty(n_sims)
    fA2 = np.empty(n_sims)
    fE = np.empty(n_sims)

    for i in range(n_sims):
        # Generate CS-constrained random alm for this l
        # a_{l,0} is real, a_{l,m} for m>0 are complex
        alm_l = np.empty(l + 1, dtype=complex)
        alm_l[0] = rng.standard_normal()  # m=0: real
        for m in range(1, l + 1):
            alm_l[m] = rng.standard_normal() + 1j * rng.standard_normal()

        fA1[i], fA2[i], fE[i] = d3_fractions_from_raw(alm_l, l, gamma)

    return fA1, fA2, fE


# =====================================================================
# MAIN ANALYSIS
# =====================================================================
def run_analysis(fits_path, lmax, n_sims, gamma_opt, outdir, map_label="SMICA"):
    """Run the full D3 extended analysis."""

    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)

    print("=" * 72)
    print(f"D3 EXTENDED ANALYSIS: {map_label}")
    print(f"  lmax = {lmax}, N_MC = {n_sims}")
    print(f"  D3 axis: (l, b) = ({D3_AXIS_L}, {D3_AXIS_B})")
    print(f"  Gamma optimization: {gamma_opt}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. LOAD MAP AND EXTRACT ALMS
    # ------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.1f}s] Loading {fits_path}...")
    sky_map = hp.read_map(fits_path, field=0)
    nside = hp.get_nside(sky_map)
    print(f"  NSIDE = {nside}, npix = {len(sky_map)}")

    # Downgrade if needed for efficiency (NSIDE=256 sufficient for lmax=150)
    if nside > 256 and lmax <= 384:
        print(f"  Downgrading to NSIDE=256 for efficiency...")
        sky_map = hp.ud_grade(sky_map, 256)
        nside = 256

    print(f"[{time.time()-t0:.1f}s] Extracting alms up to lmax={lmax}...")
    alm_gal = hp.map2alm(sky_map, lmax=lmax)
    Cl_obs = hp.alm2cl(alm_gal)
    print(f"  alm size = {len(alm_gal)}")

    # ------------------------------------------------------------------
    # 2. ROTATE TO D3 FRAME
    # ------------------------------------------------------------------
    print(f"[{time.time()-t0:.1f}s] Rotating to D3 frame...")
    alm_d3 = rotate_to_d3_frame(alm_gal, lmax)

    # Verification: at l=3, m=+-3 fraction should be ~0.94
    idx_33 = hp.Alm.getidx(lmax, 3, 3)
    total_l3 = sum(
        (1 if m == 0 else 2) * np.abs(alm_d3[hp.Alm.getidx(lmax, 3, m)])**2
        for m in range(4)
    )
    f_m3 = 2 * np.abs(alm_d3[idx_33])**2 / total_l3
    print(f"  Verification: f_{{m=+-3}}(l=3) = {f_m3:.4f} (expected ~0.94)")
    if f_m3 < 0.90:
        print("  WARNING: rotation verification failed!")

    # ------------------------------------------------------------------
    # 3. GAMMA OPTIMIZATION (optional)
    # ------------------------------------------------------------------
    gamma = 0.0
    if gamma_opt:
        print(f"[{time.time()-t0:.1f}s] Optimizing azimuthal phase gamma...")
        l_range = range(2, min(lmax, 50))
        gamma, score = optimize_gamma(alm_d3, lmax, l_range, n_steps=360)
        print(f"  Optimal gamma = {np.degrees(gamma):.1f} deg (score = {score:.4f})")
    else:
        print(f"  Using gamma = 0 (no optimization)")

    # ------------------------------------------------------------------
    # 4. COMPUTE D3 FRACTIONS FOR ALL l
    # ------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.1f}s] Computing D3 fractions for l=2..{lmax}...")

    results = []
    print(f"\n{'l':>4} {'f_A1':>8} {'f_A2':>8} {'f_E':>8} | "
          f"{'E[A2]':>8} {'dA2':>8} | {'l%3':>4}")
    print("-" * 62)

    for l in range(2, lmax + 1):
        fA1, fA2, fE = d3_fractions(alm_d3, lmax, l, gamma)
        dA1, dA2, dE, efA1, efA2, efE = d3_irrep_dimensions(l)
        delta_A2 = fA2 - efA2

        results.append({
            'l': l,
            'f_A1': fA1, 'f_A2': fA2, 'f_E': fE,
            'ef_A1': efA1, 'ef_A2': efA2, 'ef_E': efE,
            'delta_A2': delta_A2,
            'l_mod3': l % 3,
        })

        flag = " <<" if delta_A2 > 0.20 else (" <" if delta_A2 > 0.10 else "")
        if l <= 20 or l % 10 == 0 or abs(delta_A2) > 0.15:
            print(f"{l:>4} {fA1:>8.4f} {fA2:>8.4f} {fE:>8.4f} | "
                  f"{efA2:>8.4f} {delta_A2:>+8.4f} | {l%3:>4}{flag}")

    print(f"\n[{time.time()-t0:.1f}s] D3 decomposition complete for {len(results)} multipoles.")

    # ------------------------------------------------------------------
    # 5. MONTE CARLO
    # ------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.1f}s] Running Monte Carlo (N={n_sims})...")

    rng = np.random.default_rng(42)
    mc_results = {}

    for i, r in enumerate(results):
        l = r['l']
        fA1_mc, fA2_mc, fE_mc = mc_d3_fractions(l, n_sims, gamma, rng)

        # PTEs (one-sided: fraction of MC >= observed)
        pte_A2 = np.mean(fA2_mc >= r['f_A2'])
        pte_E = np.mean(fE_mc <= r['f_E'])  # one-sided: is E depleted?

        mc_results[l] = {
            'pte_A2': pte_A2,
            'mc_mean_A2': np.mean(fA2_mc),
            'mc_std_A2': np.std(fA2_mc),
            'z_A2': (r['f_A2'] - np.mean(fA2_mc)) / np.std(fA2_mc) if np.std(fA2_mc) > 0 else 0,
            'pte_E': pte_E,
        }

        # Progress
        if (i + 1) % 25 == 0:
            print(f"  [{time.time()-t0:.1f}s] Completed l=2..{l} ({i+1}/{len(results)})")

    print(f"[{time.time()-t0:.1f}s] Monte Carlo complete.")

    # ------------------------------------------------------------------
    # 6. DISPLAY MC RESULTS
    # ------------------------------------------------------------------
    print(f"\n{'l':>4} {'f_A2':>8} {'MC_mean':>8} {'MC_std':>8} {'PTE':>10} {'z':>6} | {'sig':>6}")
    print("-" * 65)

    for r in results:
        l = r['l']
        mc = mc_results[l]
        pte = mc['pte_A2']
        z = mc['z_A2']

        sig = ""
        if pte == 0:
            sig = f"<{1/n_sims:.0e}"
        elif pte < 0.001:
            sig = "***"
        elif pte < 0.01:
            sig = "**"
        elif pte < 0.05:
            sig = "*"

        pte_str = f"{pte:.4f}" if pte > 0 else f"<{1/n_sims:.0e}"

        if l <= 20 or l % 10 == 0 or pte < 0.05 or abs(r['delta_A2']) > 0.15:
            print(f"{l:>4} {r['f_A2']:>8.4f} {mc['mc_mean_A2']:>8.4f} "
                  f"{mc['mc_std_A2']:>8.4f} {pte_str:>10} {z:>+5.1f} | {sig:>6}")

    # ------------------------------------------------------------------
    # 7. COMBINED STATISTICS
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("COMBINED STATISTICS")
    print(f"{'='*72}")

    # Fisher combined PTE for various l ranges
    def fisher_combined(l_list, label):
        ptes = [mc_results[l]['pte_A2'] for l in l_list]
        # Handle PTE=0: set to 1/(2*n_sims) as upper bound
        ptes_safe = [max(p, 0.5 / n_sims) for p in ptes]
        fisher_stat = -2 * sum(np.log(p) for p in ptes_safe)
        fisher_p = chi2.sf(fisher_stat, 2 * len(ptes))
        n_sig_005 = sum(1 for p in ptes if p < 0.05)
        n_sig_001 = sum(1 for p in ptes if p < 0.01)
        print(f"\n  {label} (N={len(l_list)} multipoles):")
        print(f"    Fisher stat = {fisher_stat:.2f}, dof = {2*len(l_list)}")
        print(f"    Fisher PTE  = {fisher_p:.2e}")
        print(f"    Significant at 5%: {n_sig_005}/{len(l_list)}")
        print(f"    Significant at 1%: {n_sig_001}/{len(l_list)}")
        return fisher_p, n_sig_005, n_sig_001

    all_ls = list(range(2, lmax + 1))
    low_ls = [l for l in all_ls if l <= 15]
    high_ls = [l for l in all_ls if l > 15]
    l_no3 = [l for l in all_ls if l != 3]  # exclude octupole

    fisher_all = fisher_combined(all_ls, f"All l=2..{lmax}")
    fisher_low = fisher_combined(low_ls, "Low l (2-15)")
    fisher_high = fisher_combined(high_ls, f"High l (16-{lmax})")
    fisher_no3 = fisher_combined(l_no3, f"All except l=3 (2..{lmax}, l!=3)")

    # ------------------------------------------------------------------
    # 8. l MOD 3 SELECTION RULE
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("l MOD 3 SELECTION RULE")
    print(f"{'='*72}")

    for remainder in range(3):
        group = [r for r in results if r['l'] % 3 == remainder]
        mean_dA2 = np.mean([r['delta_A2'] for r in group])
        std_dA2 = np.std([r['delta_A2'] for r in group])
        n = len(group)
        se = std_dA2 / np.sqrt(n)
        t_stat = mean_dA2 / se if se > 0 else 0

        # Mean PTE for this group
        mean_pte = np.mean([mc_results[r['l']]['pte_A2'] for r in group])

        print(f"\n  l = {remainder} mod 3 ({n} multipoles):")
        print(f"    Mean delta_A2 = {mean_dA2:+.4f} +/- {se:.4f}")
        print(f"    t-statistic   = {t_stat:+.2f}")
        print(f"    Mean PTE(A2)  = {mean_pte:.4f}")

    # Permutation test: is l=0 mod 3 mean excess significant?
    mod3_0 = [r['delta_A2'] for r in results if r['l'] % 3 == 0]
    other = [r['delta_A2'] for r in results if r['l'] % 3 != 0]
    obs_diff = np.mean(mod3_0) - np.mean(other)

    all_deltas = [r['delta_A2'] for r in results]
    n_perm = min(n_sims, 50000)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(all_deltas)
        sample = perm[:len(mod3_0)]
        rest = perm[len(mod3_0):]
        if np.mean(sample) - np.mean(rest) >= obs_diff:
            count += 1
    pte_mod3 = count / n_perm

    print(f"\n  Selection rule test:")
    print(f"    Mean(l=0 mod 3) - Mean(others) = {obs_diff:+.4f}")
    print(f"    Permutation PTE (N={n_perm}) = {pte_mod3:.4f}")

    # ------------------------------------------------------------------
    # 9. TRANSITION SCALE
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("TRANSITION SCALE: SIGNAL PERSISTENCE")
    print(f"{'='*72}")

    # Rolling mean of delta_A2 (window=10)
    window = 10
    deltas = [r['delta_A2'] for r in results]
    print(f"\n  Rolling mean of delta_A2 (window={window}):")
    print(f"  {'l_start':>8}-{'l_end':>4} {'mean_dA2':>10} {'signal?':>10}")
    print("  " + "-" * 38)

    last_signal_l = 0
    for i in range(0, len(deltas) - window + 1, window // 2):
        chunk = deltas[i:i + window]
        l_start = results[i]['l']
        l_end = results[min(i + window - 1, len(results) - 1)]['l']
        mean_d = np.mean(chunk)
        sig = "SIGNAL" if mean_d > 0.02 else ""
        if mean_d > 0.02:
            last_signal_l = l_end
        print(f"  {l_start:>8}-{l_end:>4} {mean_d:>+10.4f} {sig:>10}")

    print(f"\n  Last window with signal (mean dA2 > 0.02): l ~ {last_signal_l}")

    # E -> A2 funnel correlation
    dA2_arr = np.array([r['delta_A2'] for r in results])
    dE_arr = np.array([r['f_E'] - r['ef_E'] for r in results])
    corr = np.corrcoef(dA2_arr, dE_arr)[0, 1]
    print(f"\n  E -> A2 funnel: r(delta_A2, delta_E) = {corr:.4f}")

    # Per-multipole conservation: dA1 + dA2 + dE = 0
    max_violation = max(
        abs((r['f_A1'] - r['ef_A1']) + r['delta_A2'] + (r['f_E'] - r['ef_E']))
        for r in results
    )
    print(f"  Conservation check: max |dA1+dA2+dE| = {max_violation:.2e}")

    # ------------------------------------------------------------------
    # 10. SAVE RESULTS
    # ------------------------------------------------------------------
    output = {
        'metadata': {
            'map_label': map_label,
            'fits_path': fits_path,
            'lmax': lmax,
            'n_sims': n_sims,
            'gamma_deg': np.degrees(gamma),
            'gamma_optimized': gamma_opt,
            'd3_axis': {'l': D3_AXIS_L, 'b': D3_AXIS_B},
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_seconds': time.time() - t0,
        },
        'per_l': {
            str(r['l']): {
                'f_A1': r['f_A1'], 'f_A2': r['f_A2'], 'f_E': r['f_E'],
                'ef_A2': r['ef_A2'], 'delta_A2': r['delta_A2'],
                'l_mod3': r['l_mod3'],
                'pte_A2': mc_results[r['l']]['pte_A2'],
                'z_A2': mc_results[r['l']]['z_A2'],
                'mc_mean_A2': mc_results[r['l']]['mc_mean_A2'],
                'mc_std_A2': mc_results[r['l']]['mc_std_A2'],
            } for r in results
        },
        'combined': {
            'fisher_all': fisher_all[0],
            'fisher_low': fisher_low[0],
            'fisher_high': fisher_high[0],
            'fisher_no_octupole': fisher_no3[0],
            'n_sig_005_all': fisher_all[1],
            'n_sig_001_all': fisher_all[2],
            'l_mod3_pte': pte_mod3,
            'l_mod3_diff': obs_diff,
            'E_A2_correlation': corr,
            'last_signal_l': last_signal_l,
        },
        'l_mod3': {
            str(rem): {
                'mean_delta_A2': float(np.mean([r['delta_A2'] for r in results if r['l']%3==rem])),
                'n_multipoles': len([r for r in results if r['l']%3==rem]),
                'mean_pte': float(np.mean([mc_results[r['l']]['pte_A2'] for r in results if r['l']%3==rem])),
            } for rem in range(3)
        },
    }

    outfile = os.path.join(outdir, f'd3_extended_{map_label.lower()}_lmax{lmax}.json')
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[{time.time()-t0:.1f}s] Results saved to {outfile}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"SUMMARY: {map_label} (lmax={lmax}, N_MC={n_sims})")
    print(f"{'='*72}")
    print(f"""
  D3 axis: (l, b) = ({D3_AXIS_L}, {D3_AXIS_B})
  Gamma: {np.degrees(gamma):.1f} deg {'(optimized)' if gamma_opt else '(fixed)'}

  FISHER COMBINED PTEs:
    All l=2..{lmax}:         {fisher_all[0]:.2e}
    Low l (2-15):          {fisher_low[0]:.2e}
    High l (16-{lmax}):       {fisher_high[0]:.2e}
    Excluding octupole:    {fisher_no3[0]:.2e}

  SIGNIFICANT MULTIPOLES:
    PTE < 0.05: {fisher_all[1]}/{len(all_ls)}
    PTE < 0.01: {fisher_all[2]}/{len(all_ls)}

  l MOD 3 SELECTION RULE:
    l=0 mod 3 mean dA2: {np.mean([r['delta_A2'] for r in results if r['l']%3==0]):+.4f}
    l=1 mod 3 mean dA2: {np.mean([r['delta_A2'] for r in results if r['l']%3==1]):+.4f}
    l=2 mod 3 mean dA2: {np.mean([r['delta_A2'] for r in results if r['l']%3==2]):+.4f}
    Permutation PTE:     {pte_mod3:.4f}

  SIGNAL PERSISTENCE:
    Last detected at l ~ {last_signal_l}
    E -> A2 correlation: r = {corr:.4f}

  Total time: {time.time()-t0:.1f}s
""")

    return output


# =====================================================================
# CROSS-MAP COMPARISON
# =====================================================================
def cross_map_compare(result_files, outdir):
    """Compare D3 results across multiple maps."""

    print("=" * 72)
    print("CROSS-MAP COMPARISON")
    print("=" * 72)

    data = {}
    for fpath in result_files:
        with open(fpath) as f:
            d = json.load(f)
        label = d['metadata']['map_label']
        data[label] = d

    labels = list(data.keys())
    if len(labels) < 2:
        print("Need at least 2 maps to compare.")
        return

    # Compare per-l f_A2
    ref = labels[0]
    all_ls = sorted(data[ref]['per_l'].keys(), key=int)

    print(f"\n{'l':>4}", end="")
    for lab in labels:
        print(f"  {lab:>10}", end="")
    print(f"  {'max_dev':>10}")
    print("-" * (14 + 12 * len(labels)))

    max_devs = []
    for l_str in all_ls:
        vals = [data[lab]['per_l'][l_str]['f_A2'] for lab in labels]
        max_dev = max(vals) - min(vals)
        max_devs.append(max_dev)

        l = int(l_str)
        if l <= 20 or l % 20 == 0 or max_dev > 0.05:
            print(f"{l:>4}", end="")
            for v in vals:
                print(f"  {v:>10.4f}", end="")
            print(f"  {max_dev:>10.4f}")

    # Cross-map correlations
    print(f"\nCross-map f_A2 correlations:")
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if j <= i:
                continue
            v1 = [data[l1]['per_l'][l_str]['f_A2'] for l_str in all_ls]
            v2 = [data[l2]['per_l'][l_str]['f_A2'] for l_str in all_ls]
            corr = np.corrcoef(v1, v2)[0, 1]
            mean_dev = np.mean(np.abs(np.array(v1) - np.array(v2)))
            print(f"  {l1} vs {l2}: r = {corr:.4f}, mean |dev| = {mean_dev:.4f}")

    # Fisher PTE comparison
    print(f"\nFisher combined PTEs:")
    print(f"  {'Map':>12} {'All':>12} {'High-l':>12} {'No-oct':>12} {'l%3 PTE':>10}")
    print("  " + "-" * 58)
    for lab in labels:
        d = data[lab]['combined']
        print(f"  {lab:>12} {d['fisher_all']:>12.2e} {d['fisher_high']:>12.2e} "
              f"{d['fisher_no_octupole']:>12.2e} {d['l_mod3_pte']:>10.4f}")

    # Consistency metric
    mean_max_dev = np.mean(max_devs)
    print(f"\n  Mean max cross-map deviation: {mean_max_dev:.4f}")
    print(f"  {'CONSISTENT' if mean_max_dev < 0.03 else 'SOME VARIATION'}")

    outfile = os.path.join(outdir, 'cross_map_comparison.json')
    comp = {
        'maps': labels,
        'mean_max_deviation': mean_max_dev,
        'fisher_ptes': {lab: data[lab]['combined'] for lab in labels},
    }
    with open(outfile, 'w') as f:
        json.dump(comp, f, indent=2)
    print(f"\n  Saved to {outfile}")


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='Extended D3 CMB Validation')
    parser.add_argument('--fits', required=True, help='Path to Planck FITS map')
    parser.add_argument('--lmax', type=int, default=150, help='Maximum multipole (default: 150)')
    parser.add_argument('--nsims', type=int, default=10000, help='MC simulations (default: 10000)')
    parser.add_argument('--gamma-opt', action='store_true', help='Optimize azimuthal phase')
    parser.add_argument('--outdir', default='extended_results', help='Output directory')
    parser.add_argument('--label', default=None, help='Map label (default: inferred from filename)')
    parser.add_argument('--compare', nargs='+', help='Compare multiple result JSON files')

    args = parser.parse_args()

    if args.compare:
        cross_map_compare(args.compare, args.outdir)
        return

    # Infer label from filename
    label = args.label
    if label is None:
        fname = os.path.basename(args.fits).lower()
        for name in ['smica', 'nilc', 'sevem', 'commander']:
            if name in fname:
                label = name.upper()
                break
        if label is None:
            label = 'UNKNOWN'

    run_analysis(args.fits, args.lmax, args.nsims, args.gamma_opt, args.outdir, label)


if __name__ == '__main__':
    main()
