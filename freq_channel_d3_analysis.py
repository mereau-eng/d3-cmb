#!/usr/bin/env python3
"""
Planck Frequency-Channel D3 Test
=================================
Tests whether the D3 = Weyl(SU(3)) CMB signal is frequency-independent
(real CMB) or frequency-dependent (foreground or systematic artifact).

Runs D3 irrep decomposition on individual Planck HFI frequency maps
(100, 143, 217 GHz), foreground tracer (353 GHz), and half-mission splits.

Usage:
  python3 freq_channels/scripts/freq_channel_d3_analysis.py \
    --datadir freq_channels/data/ \
    --outdir freq_channels/results/ \
    --smica-ref extended_results/d3_extended_smica_lmax150.json \
    --nmc 10000 --lmax 15

Author: Robert Mereau / Claude analysis extension
Date: April 2026
"""

import numpy as np
import healpy as hp
import json
import argparse
import time
import os
import sys
from scipy.stats import chi2


# =====================================================================
# D3 AXIS (from prior NSIDE=32 optimization)
# =====================================================================
D3_AXIS_L = 50.3       # Galactic longitude (degrees)
D3_AXIS_B = -64.9      # Galactic latitude (degrees)
D3_ANTI_L = 230.3      # Antipode longitude
D3_ANTI_B = 64.9       # Antipode latitude

# Beam FWHM in arcmin for each HFI frequency
BEAM_FWHM = {
    100: 9.66,
    143: 7.27,
    217: 4.99,
    353: 4.94,
}


# =====================================================================
# ROTATION: Galactic -> D3 frame
# =====================================================================
def rotate_to_d3_frame(alm, lmax):
    """Rotate alm from Galactic coordinates to D3 frame (D3 axis = Z)."""
    theta_d3 = np.radians(90.0 - D3_ANTI_B)
    phi_d3 = np.radians(D3_ANTI_L)
    alm_d3 = alm.copy()
    hp.rotate_alm(alm_d3, -phi_d3, -theta_d3, 0, lmax=lmax)
    return alm_d3


# =====================================================================
# D3 IRREP DECOMPOSITION
# =====================================================================
def d3_irrep_dimensions(l):
    """Theoretical D3 irrep subspace dimensions for multipole l."""
    dim_A1 = 1
    dim_A2 = 0
    for m in range(1, l + 1):
        if m % 3 == 0:
            dim_A1 += 1
            dim_A2 += 1
    dim_E = (2 * l + 1) - dim_A1 - dim_A2
    total = 2 * l + 1
    return dim_A1, dim_A2, dim_E, dim_A1/total, dim_A2/total, dim_E/total


def d3_fractions(alm_d3, lmax_alm, l):
    """Compute D3 irrep power fractions from alm in D3 frame."""
    total = A1 = A2 = E = 0.0
    for m in range(0, l + 1):
        idx = hp.Alm.getidx(lmax_alm, l, m)
        c = alm_d3[idx]
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


def d3_fractions_from_raw(alm_l, l):
    """D3 fractions from a raw complex vector of CS-constrained alms."""
    total = A1 = A2 = E = 0.0
    for m in range(0, l + 1):
        c = alm_l[m]
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
# MONTE CARLO
# =====================================================================
def mc_d3_fractions(l, n_sims, rng=None):
    """Generate MC distribution of D3 fractions for multipole l.

    For isotropic Gaussian random fields, the D3 fraction distribution
    is the same regardless of frame (by rotational invariance).
    """
    if rng is None:
        rng = np.random.default_rng()
    fA1 = np.empty(n_sims)
    fA2 = np.empty(n_sims)
    fE = np.empty(n_sims)
    for i in range(n_sims):
        alm_l = np.empty(l + 1, dtype=complex)
        alm_l[0] = rng.standard_normal()
        for m in range(1, l + 1):
            alm_l[m] = rng.standard_normal() + 1j * rng.standard_normal()
        fA1[i], fA2[i], fE[i] = d3_fractions_from_raw(alm_l, l)
    return fA1, fA2, fE


# =====================================================================
# MAP LOADING
# =====================================================================
def load_freq_map(filepath, nside_out=64, mask=None, is_nested=True):
    """Load a Planck HFI frequency map.

    Args:
        filepath: Path to FITS file
        nside_out: Target NSIDE for downgrade
        mask: Optional mask array at target NSIDE (1=keep, 0=mask)
        is_nested: Whether input is NESTED ordering (True for HFI maps)

    Returns:
        (sky_map, alm, nside, f_sky)
    """
    print(f"  Loading {os.path.basename(filepath)}...")
    raw = hp.read_map(filepath, field=0, nest=is_nested)
    nside_in = hp.get_nside(raw)

    # Convert NESTED to RING if needed
    if is_nested:
        raw = hp.reorder(raw, n2r=True)

    # Downgrade
    if nside_in != nside_out:
        sky_map = hp.ud_grade(raw, nside_out)
    else:
        sky_map = raw

    # Apply mask
    if mask is not None:
        sky_map = sky_map * mask
        # Set masked pixels to UNSEEN for map2alm
        sky_map[mask < 0.5] = hp.UNSEEN
        f_sky = np.mean(mask > 0.5)
    else:
        f_sky = 1.0

    return sky_map, f_sky


def load_component_sep_map(filepath, nside_out=64, mask=None):
    """Load a component-separated map (RING ordering)."""
    print(f"  Loading {os.path.basename(filepath)}...")
    raw = hp.read_map(filepath, field=0)
    nside_in = hp.get_nside(raw)

    if nside_in != nside_out:
        sky_map = hp.ud_grade(raw, nside_out)
    else:
        sky_map = raw

    if mask is not None:
        sky_map = sky_map * mask
        sky_map[mask < 0.5] = hp.UNSEEN
        f_sky = np.mean(mask > 0.5)
    else:
        f_sky = 1.0

    return sky_map, f_sky


def load_mask(datadir, nside_out=64):
    """Load and combine GAL060 Galactic mask and point source mask.

    GAL060 is field 2 in HFI_Mask_GalPlane-apo0_2048_R2.00.fits.
    Point source mask is in HFI_Mask_PointSrc_2048_R2.00.fits.
    """
    # GAL060 mask
    gal_path = os.path.join(datadir, "HFI_Mask_GalPlane-apo0_2048_R2.00.fits")
    if os.path.exists(gal_path):
        # GAL060 is field index 2 (GAL020=0, GAL040=1, GAL060=2, ...)
        gal_mask = hp.read_map(gal_path, field=2, nest=True)
        gal_mask = hp.reorder(gal_mask, n2r=True)
        # CRITICAL: convert uint8 to float64 before ud_grade (uint8 rounds to 0)
        gal_mask = gal_mask.astype(np.float64)
        print(f"  GAL060 mask loaded: f_sky = {np.mean(gal_mask > 0.5):.3f} (at native NSIDE)")
    else:
        print(f"  WARNING: GAL060 mask not found at {gal_path}, using no Galactic mask")
        gal_mask = None

    # Point source mask
    ps_path = os.path.join(datadir, "HFI_Mask_PointSrc_2048_R2.00.fits")
    if os.path.exists(ps_path):
        # Field 0 is the combined mask across all HFI frequencies
        ps_mask = hp.read_map(ps_path, field=0, nest=True)
        ps_mask = hp.reorder(ps_mask, n2r=True)
        # CRITICAL: convert uint8 to float64 before ud_grade
        ps_mask = ps_mask.astype(np.float64)
        print(f"  Point source mask loaded: f_sky = {np.mean(ps_mask > 0.5):.3f}")
    else:
        print(f"  WARNING: Point source mask not found at {ps_path}")
        ps_mask = None

    # Combine
    if gal_mask is not None and ps_mask is not None:
        combined = gal_mask * ps_mask
    elif gal_mask is not None:
        combined = gal_mask
    elif ps_mask is not None:
        combined = ps_mask
    else:
        return None, 1.0

    # Downgrade to target NSIDE
    combined_dg = hp.ud_grade(combined, nside_out)
    # Threshold at 0.5 to keep binary
    combined_dg = (combined_dg > 0.5).astype(float)
    f_sky = np.mean(combined_dg > 0.5)
    print(f"  Combined mask at NSIDE={nside_out}: f_sky = {f_sky:.3f}")

    return combined_dg, f_sky


# =====================================================================
# D3 ANALYSIS PER MAP
# =====================================================================
def run_d3_analysis(sky_map, mask, lmax, nmc, nside, label="",
                    rng_seed=42, do_axis_scramble=False, n_axes=1000):
    """Run full D3 analysis on a single map.

    Returns a dict with per-l results, Fisher PTE, parity gate, E->A2 funnel.
    """
    t0 = time.time()
    f_sky = np.mean(mask > 0.5) if mask is not None else 1.0

    # Extract alm
    alm_gal = hp.map2alm(sky_map, lmax=lmax)

    # Rotate to D3 frame
    alm_d3 = rotate_to_d3_frame(alm_gal, lmax)

    # Verification at l=3
    idx_33 = hp.Alm.getidx(lmax, 3, 3)
    total_l3 = sum(
        (1 if m == 0 else 2) * np.abs(alm_d3[hp.Alm.getidx(lmax, 3, m)])**2
        for m in range(4)
    )
    f_m3 = 2 * np.abs(alm_d3[idx_33])**2 / total_l3 if total_l3 > 0 else 0
    print(f"  [{label}] f_{{m=+-3}}(l=3) = {f_m3:.4f}")

    # Compute D3 fractions
    per_l = {}
    for l in range(2, lmax + 1):
        fA1, fA2, fE = d3_fractions(alm_d3, lmax, l)
        dA1, dA2, dE, efA1, efA2, efE = d3_irrep_dimensions(l)
        delta_A2 = fA2 - efA2
        per_l[l] = {
            'f_A1': float(fA1), 'f_A2': float(fA2), 'f_E': float(fE),
            'ef_A1': float(efA1), 'ef_A2': float(efA2), 'ef_E': float(efE),
            'delta_A2': float(delta_A2),
        }

    # Monte Carlo
    rng = np.random.default_rng(rng_seed)
    for l in range(2, lmax + 1):
        fA1_mc, fA2_mc, fE_mc = mc_d3_fractions(l, nmc, rng)
        pte_A2 = float(np.mean(fA2_mc >= per_l[l]['f_A2']))
        mc_mean = float(np.mean(fA2_mc))
        mc_std = float(np.std(fA2_mc))
        z_A2 = float((per_l[l]['f_A2'] - mc_mean) / mc_std) if mc_std > 0 else 0.0
        per_l[l]['pte_A2'] = pte_A2
        per_l[l]['z_A2'] = z_A2
        per_l[l]['mc_mean_A2'] = mc_mean
        per_l[l]['mc_std_A2'] = mc_std

    # Fisher combined PTE (l=2-15)
    low_ls = [l for l in range(2, min(lmax + 1, 16))]
    ptes = [per_l[l]['pte_A2'] for l in low_ls]
    ptes_safe = [max(p, 0.5 / nmc) for p in ptes]
    fisher_stat = -2 * sum(np.log(p) for p in ptes_safe)
    fisher_pte = float(chi2.sf(fisher_stat, 2 * len(ptes)))

    # Parity gate
    odd_sum = sum(per_l[l]['delta_A2'] for l in low_ls if l % 2 == 1)
    even_sum = sum(per_l[l]['delta_A2'] for l in low_ls if l % 2 == 0)
    if abs(odd_sum) + abs(even_sum) > 0:
        parity_ratio = (odd_sum - even_sum) / (abs(odd_sum) + abs(even_sum))
    else:
        parity_ratio = 0.0

    # E -> A2 funnel (inter-irrep correlation)
    delta_A2_vec = [per_l[l]['delta_A2'] for l in low_ls]
    delta_E_vec = [per_l[l]['f_E'] - per_l[l]['ef_E'] for l in low_ls]
    if len(delta_A2_vec) > 2:
        r_A2_E = float(np.corrcoef(delta_A2_vec, delta_E_vec)[0, 1])
    else:
        r_A2_E = 0.0

    # Axis scramble (optional)
    axis_scramble_pte = None
    if do_axis_scramble and n_axes > 0:
        rng_ax = np.random.default_rng(rng_seed + 100)
        observed_fisher = fisher_stat
        n_more_extreme = 0
        for _ in range(n_axes):
            # Random axis
            cos_theta = rng_ax.uniform(-1, 1)
            phi = rng_ax.uniform(0, 2 * np.pi)
            theta = np.arccos(cos_theta)
            alm_rand = alm_gal.copy()
            hp.rotate_alm(alm_rand, -phi, -theta, 0, lmax=lmax)

            # Compute Fisher stat at random axis
            ptes_rand = []
            for l in low_ls:
                fA1_r, fA2_r, fE_r = d3_fractions(alm_rand, lmax, l)
                _, _, _, _, efA2, _ = d3_irrep_dimensions(l)
                fA1_mc, fA2_mc, fE_mc = mc_d3_fractions(l, min(nmc, 2000), rng_ax)
                pte_r = float(np.mean(fA2_mc >= fA2_r))
                ptes_rand.append(max(pte_r, 0.5 / min(nmc, 2000)))
            fisher_rand = -2 * sum(np.log(p) for p in ptes_rand)
            if fisher_rand >= observed_fisher:
                n_more_extreme += 1
        axis_scramble_pte = float((n_more_extreme + 1) / (n_axes + 1))

    # Cumulative Xi(L)
    xi_values = []
    running = 0.0
    for l in low_ls:
        running += per_l[l]['delta_A2'] * (2 * l + 1)
        xi_values.append(float(running))

    # Driving multipoles (PTE < 0.05 and delta_A2 > 0)
    driving = [l for l in low_ls if per_l[l]['pte_A2'] < 0.05 and per_l[l]['delta_A2'] > 0]

    elapsed = time.time() - t0
    print(f"  [{label}] Fisher PTE = {fisher_pte:.4f}, parity = {parity_ratio:+.3f}, "
          f"r(A2,E) = {r_A2_E:.3f}, driving = {driving} ({elapsed:.1f}s)")

    result = {
        'label': label,
        'nside': nside,
        'lmax': lmax,
        'f_sky': float(f_sky),
        'd3_axis': {'l': D3_AXIS_L, 'b': D3_AXIS_B},
        'per_l': {str(l): per_l[l] for l in per_l},
        'fisher_pte_low_l': fisher_pte,
        'fisher_stat_low_l': float(fisher_stat),
        'n_mc': nmc,
        'parity_gate': {
            'odd_l_sum_delta_A2': float(odd_sum),
            'even_l_sum_delta_A2': float(even_sum),
            'parity_ratio': float(parity_ratio),
        },
        'inter_irrep_r': r_A2_E,
        'xi_cumulative': xi_values,
        'driving_multipoles': driving,
        'f_m3_l3': float(f_m3),
    }

    if axis_scramble_pte is not None:
        result['axis_scramble_pte'] = axis_scramble_pte

    return result


# =====================================================================
# CROSS-FREQUENCY CONSISTENCY
# =====================================================================
def cross_frequency_analysis(results_dict, smica_ref=None):
    """Compute cross-frequency consistency statistics.

    Args:
        results_dict: {freq_label: result_dict} for each frequency
        smica_ref: Optional SMICA reference result

    Returns:
        Consistency statistics dict
    """
    freqs = sorted(results_dict.keys())
    lmax = 15

    # Per-l spread
    per_l_spread = {}
    for l in range(2, lmax + 1):
        ls = str(l)
        vals = {}
        for freq in freqs:
            r = results_dict[freq]
            if ls in r['per_l']:
                vals[freq] = r['per_l'][ls]['f_A2']
        if len(vals) >= 2:
            spread = max(vals.values()) - min(vals.values())
        else:
            spread = 0.0
        entry = {f"f_A2_{freq}": float(v) for freq, v in vals.items()}
        entry['spread'] = float(spread)
        per_l_spread[ls] = entry

    # Cross-frequency correlations of delta_A2 vectors
    delta_A2_vecs = {}
    for freq in freqs:
        r = results_dict[freq]
        vec = [r['per_l'][str(l)]['delta_A2'] for l in range(2, lmax + 1)]
        delta_A2_vecs[freq] = np.array(vec)

    cross_corr = {}
    for i, f1 in enumerate(freqs):
        for f2 in freqs[i+1:]:
            r = float(np.corrcoef(delta_A2_vecs[f1], delta_A2_vecs[f2])[0, 1])
            cross_corr[f"r_{f1}_{f2}"] = r

    # Fisher PTE comparison
    fisher_comparison = {freq: results_dict[freq]['fisher_pte_low_l'] for freq in freqs}
    if smica_ref is not None:
        fisher_comparison['SMICA'] = smica_ref

    # f-vector cosine similarity across frequencies
    f_vec_cosines = {}
    for l in range(2, lmax + 1):
        ls = str(l)
        vecs = {}
        for freq in freqs:
            r = results_dict[freq]
            if ls in r['per_l']:
                vecs[freq] = np.array([
                    r['per_l'][ls]['f_A1'],
                    r['per_l'][ls]['f_A2'],
                    r['per_l'][ls]['f_E']
                ])
        cosines = {}
        flist = list(vecs.keys())
        for i, f1 in enumerate(flist):
            for f2 in flist[i+1:]:
                v1, v2 = vecs[f1], vecs[f2]
                cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30))
                cosines[f"{f1}_{f2}"] = cos
        f_vec_cosines[ls] = cosines

    # Mean f-vector cosine across all l
    all_cosines = []
    for ls in f_vec_cosines:
        all_cosines.extend(f_vec_cosines[ls].values())
    mean_cosine = float(np.mean(all_cosines)) if all_cosines else 0.0

    return {
        'test': 'frequency_consistency',
        'frequencies': freqs,
        'mask': 'GAL060_x_PointSrc',
        'per_l_spread': per_l_spread,
        'cross_freq_correlations': cross_corr,
        'fisher_pte_comparison': fisher_comparison,
        'f_vec_cosines_per_l': f_vec_cosines,
        'mean_f_vec_cosine': mean_cosine,
    }


# =====================================================================
# HALF-MISSION CONSISTENCY
# =====================================================================
def halfmission_analysis(hm1_result, hm2_result, diff_result=None, label=""):
    """Compare half-mission results.

    Args:
        hm1_result: D3 result for half-mission 1
        hm2_result: D3 result for half-mission 2
        diff_result: D3 result for (HM1-HM2)/2 difference map
        label: e.g. "143GHz" or "SMICA"

    Returns:
        Consistency dict
    """
    lmax = 15

    # Per-l f_A2 correlation
    fa2_hm1 = [hm1_result['per_l'][str(l)]['f_A2'] for l in range(2, lmax + 1)]
    fa2_hm2 = [hm2_result['per_l'][str(l)]['f_A2'] for l in range(2, lmax + 1)]
    hm_corr = float(np.corrcoef(fa2_hm1, fa2_hm2)[0, 1])

    # Delta-A2 correlation
    da2_hm1 = [hm1_result['per_l'][str(l)]['delta_A2'] for l in range(2, lmax + 1)]
    da2_hm2 = [hm2_result['per_l'][str(l)]['delta_A2'] for l in range(2, lmax + 1)]
    da2_corr = float(np.corrcoef(da2_hm1, da2_hm2)[0, 1])

    # Per-l f-vector cosine
    cosines = []
    for l in range(2, lmax + 1):
        ls = str(l)
        v1 = np.array([hm1_result['per_l'][ls]['f_A1'],
                        hm1_result['per_l'][ls]['f_A2'],
                        hm1_result['per_l'][ls]['f_E']])
        v2 = np.array([hm2_result['per_l'][ls]['f_A1'],
                        hm2_result['per_l'][ls]['f_A2'],
                        hm2_result['per_l'][ls]['f_E']])
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30))
        cosines.append(cos)
    mean_cosine = float(np.mean(cosines))

    entry = {
        'label': label,
        'hm1_fisher_pte': hm1_result['fisher_pte_low_l'],
        'hm2_fisher_pte': hm2_result['fisher_pte_low_l'],
        'hm1_hm2_f_A2_correlation': hm_corr,
        'hm1_hm2_delta_A2_correlation': da2_corr,
        'hm1_hm2_f_vec_mean_cosine': mean_cosine,
        'hm1_parity_ratio': hm1_result['parity_gate']['parity_ratio'],
        'hm2_parity_ratio': hm2_result['parity_gate']['parity_ratio'],
    }

    if diff_result is not None:
        entry['diff_map_fisher_pte'] = diff_result['fisher_pte_low_l']
        entry['diff_map_parity_ratio'] = diff_result['parity_gate']['parity_ratio']

    return entry


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Planck Frequency-Channel D3 Test")
    parser.add_argument('--datadir', required=True, help='Directory with FITS files')
    parser.add_argument('--outdir', required=True, help='Output directory')
    parser.add_argument('--smica-ref', default=None, help='SMICA reference JSON')
    parser.add_argument('--nmc', type=int, default=10000, help='Number of MC sims')
    parser.add_argument('--naxes', type=int, default=1000, help='Axis scramble directions')
    parser.add_argument('--lmax', type=int, default=15, help='Maximum multipole')
    parser.add_argument('--nside', type=int, default=64, help='Target NSIDE')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()

    print("=" * 72)
    print("PLANCK FREQUENCY-CHANNEL D3 TEST")
    print(f"  lmax = {args.lmax}, NSIDE = {args.nside}, N_MC = {args.nmc}")
    print(f"  D3 axis: (l, b) = ({D3_AXIS_L}, {D3_AXIS_B})")
    print("=" * 72)

    # Load SMICA reference
    smica_fisher = None
    if args.smica_ref and os.path.exists(args.smica_ref):
        with open(args.smica_ref) as f:
            smica_data = json.load(f)
        # Navigate nested structure
        combined = smica_data.get('combined', {})
        smica_fisher = combined.get('fisher_low',
                                    smica_data.get('fisher_pte_low_l',
                                                   smica_data.get('fisher_combined_pte_l2_15')))
        print(f"  SMICA reference Fisher PTE: {smica_fisher}")

    # ================================================================
    # STEP 1: Load masks
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 1: LOADING MASKS")
    print(f"{'='*72}")

    mask, f_sky = load_mask(args.datadir, args.nside)
    if mask is None:
        print("  No mask available -- proceeding without masking")
        print("  WARNING: frequency maps without masking will have significant foreground contamination!")

    # ================================================================
    # STEP 2: Full-mission frequency maps (100, 143, 217 GHz)
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 2: FULL-MISSION FREQUENCY MAPS")
    print(f"{'='*72}")

    freq_results = {}
    for freq in [100, 143, 217]:
        fname = f"HFI_SkyMap_{freq}_2048_R3.01_full.fits"
        fpath = os.path.join(args.datadir, fname)
        if not os.path.exists(fpath):
            print(f"\n  SKIP: {fname} not found")
            continue

        print(f"\n--- {freq} GHz (full mission) ---")
        sky_map, fsky = load_freq_map(fpath, args.nside, mask, is_nested=True)

        result = run_d3_analysis(
            sky_map, mask, args.lmax, args.nmc, args.nside,
            label=f"{freq}GHz_full", rng_seed=42 + freq,
            do_axis_scramble=(freq == 143),  # scramble only for anchor frequency
            n_axes=args.naxes
        )
        result['experiment'] = 'Planck_PR3'
        result['map'] = f'{freq}GHz_full'
        result['frequency_GHz'] = freq
        result['data_split'] = 'full'
        result['mask'] = 'GAL060_x_PointSrc'
        result['beam_fwhm_arcmin'] = BEAM_FWHM.get(freq, 0)
        result['beam_deconvolved'] = False

        outpath = os.path.join(args.outdir, f"d3_{freq}GHz_full.json")
        with open(outpath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {outpath}")

        freq_results[f"{freq}GHz"] = result

    # ================================================================
    # STEP 3: Foreground tracer (353 GHz)
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 3: FOREGROUND TRACER (353 GHz)")
    print(f"{'='*72}")

    fname_353 = "HFI_SkyMap_353_2048_R3.01_full.fits"
    fpath_353 = os.path.join(args.datadir, fname_353)
    if os.path.exists(fpath_353):
        print(f"\n--- 353 GHz (foreground tracer) ---")
        sky_353, fsky_353 = load_freq_map(fpath_353, args.nside, mask, is_nested=True)

        result_353 = run_d3_analysis(
            sky_353, mask, args.lmax, args.nmc, args.nside,
            label="353GHz_full", rng_seed=42 + 353
        )
        result_353['experiment'] = 'Planck_PR3'
        result_353['map'] = '353GHz_full'
        result_353['frequency_GHz'] = 353
        result_353['data_split'] = 'full'
        result_353['mask'] = 'GAL060_x_PointSrc'
        result_353['beam_fwhm_arcmin'] = BEAM_FWHM.get(353, 0)
        result_353['beam_deconvolved'] = False
        result_353['note'] = 'Heavily dust-contaminated -- used as foreground tracer'

        outpath = os.path.join(args.outdir, "d3_353GHz_full.json")
        with open(outpath, 'w') as f:
            json.dump(result_353, f, indent=2)
        print(f"  Saved: {outpath}")

        freq_results['353GHz'] = result_353
    else:
        print(f"  SKIP: {fname_353} not found")

    # ================================================================
    # STEP 4: Cross-frequency consistency
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 4: CROSS-FREQUENCY CONSISTENCY")
    print(f"{'='*72}")

    # Use only the CMB frequencies (100, 143, 217) for consistency
    cmb_results = {k: v for k, v in freq_results.items() if k in ['100GHz', '143GHz', '217GHz']}
    if len(cmb_results) >= 2:
        consistency = cross_frequency_analysis(cmb_results, smica_fisher)

        # Add foreground tracer results if available
        if '353GHz' in freq_results:
            consistency['foreground_tracer_353GHz_fisher_pte'] = freq_results['353GHz']['fisher_pte_low_l']
            consistency['foreground_tracer_353GHz_driving'] = freq_results['353GHz']['driving_multipoles']

        # Verdict
        corr_vals = list(consistency['cross_freq_correlations'].values())
        mean_corr = np.mean(corr_vals) if corr_vals else 0
        if mean_corr > 0.8:
            consistency['verdict'] = 'frequency-independent (real CMB signal)'
        elif mean_corr > 0.5:
            consistency['verdict'] = 'suggestive frequency independence (moderate correlation)'
        else:
            consistency['verdict'] = 'frequency-dependent (possible foreground contamination)'

        # Check for dust gradient (217 vs 100)
        if '100GHz' in freq_results and '217GHz' in freq_results:
            f100 = freq_results['100GHz']['fisher_pte_low_l']
            f217 = freq_results['217GHz']['fisher_pte_low_l']
            consistency['dust_gradient_check'] = {
                'fisher_100GHz': f100,
                'fisher_217GHz': f217,
                'note': '217 < 100 would suggest dust contamination' if f217 < f100
                        else '100 <= 217 -- no dust gradient detected'
            }

        outpath = os.path.join(args.outdir, "freq_consistency.json")
        with open(outpath, 'w') as f:
            json.dump(consistency, f, indent=2)
        print(f"  Saved: {outpath}")

        # Print summary
        print(f"\n  Cross-frequency correlations (delta_A2 vectors):")
        for k, v in consistency['cross_freq_correlations'].items():
            print(f"    {k}: {v:.3f}")
        print(f"  Mean f-vector cosine: {consistency['mean_f_vec_cosine']:.3f}")
        print(f"  Verdict: {consistency['verdict']}")
    else:
        print("  SKIP: Need at least 2 frequency maps for consistency test")

    # ================================================================
    # STEP 5: Half-mission splits
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 5: HALF-MISSION SPLITS")
    print(f"{'='*72}")

    hm_consistency = {'test': 'halfmission_consistency', 'per_frequency': {}}

    # 5a. Frequency half-mission splits (priority: 143 GHz)
    for freq in [143, 100, 217]:
        hm1_name = f"HFI_SkyMap_{freq}_2048_R3.01_halfmission-1.fits"
        hm2_name = f"HFI_SkyMap_{freq}_2048_R3.01_halfmission-2.fits"
        hm1_path = os.path.join(args.datadir, hm1_name)
        hm2_path = os.path.join(args.datadir, hm2_name)

        if not os.path.exists(hm1_path) or not os.path.exists(hm2_path):
            print(f"\n  SKIP: {freq} GHz half-mission files not found")
            continue

        print(f"\n--- {freq} GHz half-mission splits ---")

        # HM1
        sky_hm1, _ = load_freq_map(hm1_path, args.nside, mask, is_nested=True)
        result_hm1 = run_d3_analysis(
            sky_hm1, mask, args.lmax, args.nmc, args.nside,
            label=f"{freq}GHz_HM1", rng_seed=42 + freq + 1000
        )
        result_hm1['experiment'] = 'Planck_PR3'
        result_hm1['map'] = f'{freq}GHz_hm1'
        result_hm1['frequency_GHz'] = freq
        result_hm1['data_split'] = 'halfmission-1'
        outpath = os.path.join(args.outdir, f"d3_{freq}GHz_hm1.json")
        with open(outpath, 'w') as f:
            json.dump(result_hm1, f, indent=2)

        # HM2
        sky_hm2, _ = load_freq_map(hm2_path, args.nside, mask, is_nested=True)
        result_hm2 = run_d3_analysis(
            sky_hm2, mask, args.lmax, args.nmc, args.nside,
            label=f"{freq}GHz_HM2", rng_seed=42 + freq + 2000
        )
        result_hm2['experiment'] = 'Planck_PR3'
        result_hm2['map'] = f'{freq}GHz_hm2'
        result_hm2['frequency_GHz'] = freq
        result_hm2['data_split'] = 'halfmission-2'
        outpath = os.path.join(args.outdir, f"d3_{freq}GHz_hm2.json")
        with open(outpath, 'w') as f:
            json.dump(result_hm2, f, indent=2)

        # Difference map null test: (HM1 - HM2) / 2
        print(f"  Computing (HM1 - HM2)/2 null test...")
        # Re-load without mask application for clean subtraction
        sky_hm1_raw, _ = load_freq_map(hm1_path, args.nside, mask=None, is_nested=True)
        sky_hm2_raw, _ = load_freq_map(hm2_path, args.nside, mask=None, is_nested=True)
        diff_map = (sky_hm1_raw - sky_hm2_raw) / 2.0
        if mask is not None:
            diff_map = diff_map * mask
            diff_map[mask < 0.5] = hp.UNSEEN

        result_diff = run_d3_analysis(
            diff_map, mask, args.lmax, args.nmc, args.nside,
            label=f"{freq}GHz_HM_diff", rng_seed=42 + freq + 3000
        )
        result_diff['experiment'] = 'Planck_PR3'
        result_diff['map'] = f'{freq}GHz_hm_diff'
        result_diff['frequency_GHz'] = freq
        result_diff['data_split'] = '(hm1-hm2)/2'
        result_diff['note'] = 'Null test: should be pure noise if signal is stable'
        outpath = os.path.join(args.outdir, f"d3_{freq}GHz_hm_diff.json")
        with open(outpath, 'w') as f:
            json.dump(result_diff, f, indent=2)

        # HM consistency
        hm_entry = halfmission_analysis(result_hm1, result_hm2, result_diff, f"{freq}GHz")
        hm_consistency['per_frequency'][f"{freq}GHz"] = hm_entry
        print(f"  HM correlation (f_A2): {hm_entry['hm1_hm2_f_A2_correlation']:.3f}")
        print(f"  Diff map Fisher PTE: {hm_entry['diff_map_fisher_pte']:.4f}")

    # 5b. SMICA half-mission
    smica_hm1_path = os.path.join(args.datadir, "COM_CMB_IQU-smica_2048_R3.00_hm1.fits")
    smica_hm2_path = os.path.join(args.datadir, "COM_CMB_IQU-smica_2048_R3.00_hm2.fits")
    if os.path.exists(smica_hm1_path) and os.path.exists(smica_hm2_path):
        print(f"\n--- SMICA half-mission splits ---")

        sky_shm1, _ = load_component_sep_map(smica_hm1_path, args.nside, mask)
        result_shm1 = run_d3_analysis(
            sky_shm1, mask, args.lmax, args.nmc, args.nside,
            label="SMICA_HM1", rng_seed=42 + 9001
        )
        result_shm1['experiment'] = 'Planck_PR3'
        result_shm1['map'] = 'SMICA_hm1'
        result_shm1['data_split'] = 'halfmission-1'
        outpath = os.path.join(args.outdir, "d3_smica_hm1.json")
        with open(outpath, 'w') as f:
            json.dump(result_shm1, f, indent=2)

        sky_shm2, _ = load_component_sep_map(smica_hm2_path, args.nside, mask)
        result_shm2 = run_d3_analysis(
            sky_shm2, mask, args.lmax, args.nmc, args.nside,
            label="SMICA_HM2", rng_seed=42 + 9002
        )
        result_shm2['experiment'] = 'Planck_PR3'
        result_shm2['map'] = 'SMICA_hm2'
        result_shm2['data_split'] = 'halfmission-2'
        outpath = os.path.join(args.outdir, "d3_smica_hm2.json")
        with open(outpath, 'w') as f:
            json.dump(result_shm2, f, indent=2)

        smica_hm_entry = halfmission_analysis(result_shm1, result_shm2, label="SMICA")
        hm_consistency['smica_hm'] = smica_hm_entry
        print(f"  SMICA HM correlation (f_A2): {smica_hm_entry['hm1_hm2_f_A2_correlation']:.3f}")
    else:
        print(f"\n  SKIP: SMICA half-mission files not found")

    # HM verdict
    all_corrs = []
    for freq_key, entry in hm_consistency['per_frequency'].items():
        all_corrs.append(entry['hm1_hm2_f_A2_correlation'])
    if 'smica_hm' in hm_consistency:
        all_corrs.append(hm_consistency['smica_hm']['hm1_hm2_f_A2_correlation'])

    all_diff_ptes = []
    for freq_key, entry in hm_consistency['per_frequency'].items():
        if 'diff_map_fisher_pte' in entry:
            all_diff_ptes.append(entry['diff_map_fisher_pte'])

    if all_corrs:
        mean_hm_corr = np.mean(all_corrs)
        if mean_hm_corr > 0.8 and all(p > 0.05 for p in all_diff_ptes):
            hm_consistency['verdict'] = 'time-stable'
        elif mean_hm_corr > 0.5:
            hm_consistency['verdict'] = 'marginally time-stable'
        else:
            hm_consistency['verdict'] = 'time-variable (WARNING)'
    else:
        hm_consistency['verdict'] = 'insufficient data'

    outpath = os.path.join(args.outdir, "halfmission_consistency.json")
    with open(outpath, 'w') as f:
        json.dump(hm_consistency, f, indent=2)
    print(f"\n  HM verdict: {hm_consistency['verdict']}")
    print(f"  Saved: {outpath}")

    # ================================================================
    # STEP 6: Parity gate per frequency
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 6: PARITY GATE SUMMARY")
    print(f"{'='*72}")

    print(f"\n  {'Map':<20} {'Odd-l sum':>12} {'Even-l sum':>12} {'Ratio':>10}")
    print(f"  {'-'*56}")
    for key in sorted(freq_results.keys()):
        r = freq_results[key]
        pg = r['parity_gate']
        print(f"  {key:<20} {pg['odd_l_sum_delta_A2']:>+12.4f} "
              f"{pg['even_l_sum_delta_A2']:>+12.4f} {pg['parity_ratio']:>+10.3f}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'='*72}")
    print("FINAL SUMMARY")
    print(f"{'='*72}")

    print(f"\n  {'Map':<20} {'Fisher PTE':>12} {'Parity':>10} {'r(A2,E)':>10} {'Driving':>20}")
    print(f"  {'-'*74}")
    for key in sorted(freq_results.keys()):
        r = freq_results[key]
        driving = ','.join(str(l) for l in r['driving_multipoles'])
        print(f"  {key:<20} {r['fisher_pte_low_l']:>12.4f} "
              f"{r['parity_gate']['parity_ratio']:>+10.3f} "
              f"{r['inter_irrep_r']:>10.3f} {driving:>20}")

    if smica_fisher is not None:
        print(f"  {'SMICA (ref)':<20} {smica_fisher:>12.4f}")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n{'='*72}")
    print("DONE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
