#!/usr/bin/env python3
"""
Mask-Variation Robustness Test
===============================
Tests whether the D3 signal is stable across different masking choices
or whether it is an artifact of mask geometry / foreground residuals.

Runs D3 decomposition with 13 mask levels on SMICA, then cross-map
tests at UT78 and GAL040, plus axis stability, apodization, and
hemisphere splits.

CRITICAL: MC simulations apply the same mask as the data.

Usage:
  python3 mask_variation/scripts/mask_variation_analysis.py \
    --mapdir /path/to/planck/maps/ \
    --maskdir mask_variation/masks/ \
    --galmaskfile freq_channels/data/HFI_Mask_GalPlane-apo0_2048_R2.00.fits \
    --outdir mask_variation/results/ \
    --nmc 5000 --lmax 15

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
# D3 AXIS
# =====================================================================
D3_AXIS_L = 50.3
D3_AXIS_B = -64.9
D3_ANTI_L = 230.3
D3_ANTI_B = 64.9

# Planck best-fit power spectrum (fiducial for MC)
# We generate this once and reuse
_CL_FIDS = None


def get_fiducial_cl(lmax=15):
    """Return fiducial Cl for MC (simple analytic model, normalized)."""
    global _CL_FIDS
    if _CL_FIDS is not None and len(_CL_FIDS) > lmax:
        return _CL_FIDS
    # Use simple 1/(l*(l+1)) scaling (adequate for l<=15)
    cl = np.zeros(lmax + 1)
    for l in range(2, lmax + 1):
        cl[l] = 1.0 / (l * (l + 1))
    _CL_FIDS = cl
    return cl


# =====================================================================
# ROTATION
# =====================================================================
def rotate_to_d3_frame(alm, lmax):
    """Rotate alm from Galactic coordinates to D3 frame."""
    theta_d3 = np.radians(90.0 - D3_ANTI_B)
    phi_d3 = np.radians(D3_ANTI_L)
    alm_d3 = alm.copy()
    hp.rotate_alm(alm_d3, -phi_d3, -theta_d3, 0, lmax=lmax)
    return alm_d3


def rotate_to_axis(alm, lmax, gal_l, gal_b):
    """Rotate alm to place arbitrary axis at the north pole."""
    anti_l = (gal_l + 180.0) % 360.0
    anti_b = -gal_b
    theta = np.radians(90.0 - anti_b)
    phi = np.radians(anti_l)
    alm_rot = alm.copy()
    hp.rotate_alm(alm_rot, -phi, -theta, 0, lmax=lmax)
    return alm_rot


# =====================================================================
# D3 DECOMPOSITION
# =====================================================================
def d3_irrep_dimensions(l):
    """Theoretical D3 irrep dimensions and expected fractions."""
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
    """D3 fractions from raw complex vector."""
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
# MASK CONSTRUCTION
# =====================================================================
def make_latitude_mask(nside, min_abs_b):
    """Construct a latitude cut mask: |b| > min_abs_b degrees."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    b_gal = 90.0 - np.degrees(theta)
    mask = np.zeros(npix, dtype=np.float64)
    mask[np.abs(b_gal) > min_abs_b] = 1.0
    return mask


def make_hemisphere_mask(nside, hemisphere, coord='galactic'):
    """Construct hemisphere mask.

    hemisphere: 'north' or 'south'
    coord: 'galactic' or 'ecliptic'
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    if coord == 'galactic':
        b = 90.0 - np.degrees(theta)  # Galactic latitude
    elif coord == 'ecliptic':
        # Convert Galactic to ecliptic using healpy rotation
        r = hp.Rotator(coord=['G', 'E'])
        theta_ecl, phi_ecl = r(theta, phi)
        b = 90.0 - np.degrees(theta_ecl)  # Ecliptic latitude
    else:
        raise ValueError(f"Unknown coord: {coord}")

    mask = np.zeros(npix, dtype=np.float64)
    if hemisphere == 'north':
        mask[b > 0] = 1.0
    else:
        mask[b < 0] = 1.0
    return mask


def apodize_mask(mask, fwhm_deg=5.0):
    """Apodize a binary mask with Gaussian smoothing."""
    mask_apod = hp.smoothing(mask.astype(np.float64), fwhm=np.radians(fwhm_deg))
    mask_apod = np.clip(mask_apod, 0.0, 1.0)
    return mask_apod


def load_planck_mask(filepath, field=0, nside_out=64):
    """Load a Planck mask, handle NESTED ordering and dtype."""
    try:
        mask = hp.read_map(filepath, field=field, nest=True)
        mask = hp.reorder(mask, n2r=True)
    except Exception:
        # Try RING ordering
        mask = hp.read_map(filepath, field=field)
    mask = mask.astype(np.float64)
    if hp.get_nside(mask) != nside_out:
        mask = hp.ud_grade(mask, nside_out)
    mask = (mask > 0.5).astype(np.float64)
    return mask


def load_confidence_mask(cmb_fits, nside_out=64):
    """Load pipeline confidence mask from CMB FITS (field=3 = TMASK)."""
    try:
        mask = hp.read_map(cmb_fits, field=3)
    except Exception:
        print(f"  WARNING: Could not read confidence mask from {cmb_fits}")
        return None
    mask = mask.astype(np.float64)
    if hp.get_nside(mask) != nside_out:
        mask = hp.ud_grade(mask, nside_out)
    mask = (mask > 0.5).astype(np.float64)
    return mask


# =====================================================================
# MASKED MC SIMULATIONS
# =====================================================================
def masked_mc_d3(mask, nside, lmax, n_sims, rng=None):
    """Run MC simulations with mask applied.

    For each sim:
      1. Generate Gaussian alm from fiducial Cl
      2. Synthesize map at nside
      3. Apply mask
      4. Extract pseudo-alm
      5. Rotate to D3 frame
      6. Compute D3 fractions

    Returns dict of {l: (fA1_mc, fA2_mc, fE_mc)} arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    cl = get_fiducial_cl(lmax)
    has_mask = mask is not None and np.any(mask < 0.5)

    # Pre-allocate
    mc_data = {}
    for l in range(2, lmax + 1):
        mc_data[l] = (np.empty(n_sims), np.empty(n_sims), np.empty(n_sims))

    for i in range(n_sims):
        if has_mask:
            # Full masked MC pipeline
            alm_sim = hp.synalm(cl, lmax=lmax, new=True)
            map_sim = hp.alm2map(alm_sim, nside, lmax=lmax, verbose=False)
            map_sim *= mask
            map_sim[mask < 0.5] = 0.0
            alm_pseudo = hp.map2alm(map_sim, lmax=lmax)
            alm_d3 = rotate_to_d3_frame(alm_pseudo, lmax)

            for l in range(2, lmax + 1):
                fA1, fA2, fE = d3_fractions(alm_d3, lmax, l)
                mc_data[l][0][i] = fA1
                mc_data[l][1][i] = fA2
                mc_data[l][2][i] = fE
        else:
            # No mask: use fast per-l MC (rotational invariance)
            for l in range(2, lmax + 1):
                alm_l = np.empty(l + 1, dtype=complex)
                alm_l[0] = rng.standard_normal()
                for m in range(1, l + 1):
                    alm_l[m] = rng.standard_normal() + 1j * rng.standard_normal()
                fA1, fA2, fE = d3_fractions_from_raw(alm_l, l)
                mc_data[l][0][i] = fA1
                mc_data[l][1][i] = fA2
                mc_data[l][2][i] = fE

    return mc_data


# =====================================================================
# D3 ANALYSIS WITH MASK
# =====================================================================
def run_d3_masked(sky_map, mask, lmax, nmc, nside, label="",
                  rng_seed=42, mc_data=None):
    """Run D3 analysis on a map with a given mask.

    If mc_data is provided, reuse it (saves time for same mask, different map).
    """
    t0 = time.time()

    # Apply mask
    if mask is not None:
        masked_map = sky_map.copy()
        masked_map *= mask
        masked_map[mask < 0.5] = 0.0
        f_sky = float(np.mean(mask > 0.5))
    else:
        masked_map = sky_map.copy()
        f_sky = 1.0

    # Extract alm
    alm_gal = hp.map2alm(masked_map, lmax=lmax)
    alm_d3 = rotate_to_d3_frame(alm_gal, lmax)

    # Verification
    idx_33 = hp.Alm.getidx(lmax, 3, 3)
    total_l3 = sum(
        (1 if m == 0 else 2) * np.abs(alm_d3[hp.Alm.getidx(lmax, 3, m)])**2
        for m in range(4)
    )
    f_m3 = 2 * np.abs(alm_d3[idx_33])**2 / total_l3 if total_l3 > 0 else 0

    # Compute D3 fractions
    per_l = {}
    for l in range(2, lmax + 1):
        fA1, fA2, fE = d3_fractions(alm_d3, lmax, l)
        _, _, _, efA1, efA2, efE = d3_irrep_dimensions(l)
        per_l[l] = {
            'f_A1': float(fA1), 'f_A2': float(fA2), 'f_E': float(fE),
            'ef_A1': float(efA1), 'ef_A2': float(efA2), 'ef_E': float(efE),
            'delta_A2': float(fA2 - efA2),
        }

    # MC (masked)
    if mc_data is None:
        rng = np.random.default_rng(rng_seed)
        mc_data = masked_mc_d3(mask, nside, lmax, nmc, rng)

    for l in range(2, lmax + 1):
        fA2_mc = mc_data[l][1]
        pte_A2 = float(np.mean(fA2_mc >= per_l[l]['f_A2']))
        mc_mean = float(np.mean(fA2_mc))
        mc_std = float(np.std(fA2_mc))
        z_A2 = float((per_l[l]['f_A2'] - mc_mean) / mc_std) if mc_std > 0 else 0.0
        per_l[l]['pte_A2'] = pte_A2
        per_l[l]['z_A2'] = z_A2

    # Fisher PTE
    low_ls = list(range(2, min(lmax + 1, 16)))
    ptes = [per_l[l]['pte_A2'] for l in low_ls]
    ptes_safe = [max(p, 0.5 / nmc) for p in ptes]
    fisher_stat = -2 * sum(np.log(p) for p in ptes_safe)
    fisher_pte = float(chi2.sf(fisher_stat, 2 * len(ptes)))

    # Parity gate
    odd_sum = sum(per_l[l]['delta_A2'] for l in low_ls if l % 2 == 1)
    even_sum = sum(per_l[l]['delta_A2'] for l in low_ls if l % 2 == 0)
    denom = abs(odd_sum) + abs(even_sum)
    parity_ratio = (odd_sum - even_sum) / denom if denom > 0 else 0.0

    # E -> A2 funnel
    delta_A2_vec = [per_l[l]['delta_A2'] for l in low_ls]
    delta_E_vec = [per_l[l]['f_E'] - per_l[l]['ef_E'] for l in low_ls]
    if len(delta_A2_vec) > 2 and np.std(delta_A2_vec) > 0 and np.std(delta_E_vec) > 0:
        r_A2_E = float(np.corrcoef(delta_A2_vec, delta_E_vec)[0, 1])
    else:
        r_A2_E = 0.0

    # Driving multipoles
    driving = [l for l in low_ls if per_l[l]['pte_A2'] < 0.05 and per_l[l]['delta_A2'] > 0]

    elapsed = time.time() - t0
    print(f"  [{label}] Fisher={fisher_pte:.4f}, parity={parity_ratio:+.3f}, "
          f"r(A2,E)={r_A2_E:.3f}, f_m3={f_m3:.3f}, f_sky={f_sky:.3f} ({elapsed:.1f}s)")

    result = {
        'label': label,
        'nside': nside,
        'lmax': lmax,
        'f_sky': f_sky,
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
        'driving_multipoles': driving,
        'f_m3_l3': float(f_m3),
    }

    return result, mc_data, alm_gal


# =====================================================================
# AXIS RE-OPTIMIZATION
# =====================================================================
def axis_search_masked(alm_gal, lmax, nside_grid=16):
    """Find axis maximizing f_A2(l=3) over NSIDE grid."""
    npix = hp.nside2npix(nside_grid)
    best_fA2 = -1
    best_l, best_b = D3_AXIS_L, D3_AXIS_B

    for ipix in range(npix):
        theta, phi = hp.pix2ang(nside_grid, ipix)
        gl = np.degrees(phi)
        gb = 90.0 - np.degrees(theta)

        alm_rot = rotate_to_axis(alm_gal, lmax, gl, gb)
        _, fA2, _ = d3_fractions(alm_rot, lmax, 3)

        if fA2 > best_fA2:
            best_fA2 = fA2
            best_l = gl
            best_b = gb

    # Angular separation from D3 axis
    from math import radians, sin, cos, acos
    lat1, lon1 = radians(D3_AXIS_B), radians(D3_AXIS_L)
    lat2, lon2 = radians(best_b), radians(best_l)
    cos_sep = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon1 - lon2)
    cos_sep = max(-1.0, min(1.0, cos_sep))
    sep_deg = np.degrees(acos(cos_sep))

    return best_l, best_b, best_fA2, sep_deg


# =====================================================================
# CROSS-MAP COSINE
# =====================================================================
def pairwise_cosines(results_dict, lmax=15):
    """Compute pairwise cosine similarities of delta_A2 vectors."""
    labels = sorted(results_dict.keys())
    cosines = {}

    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            v1 = np.array([results_dict[l1]['per_l'][str(l)]['delta_A2']
                          for l in range(2, lmax + 1)])
            v2 = np.array([results_dict[l2]['per_l'][str(l)]['delta_A2']
                          for l in range(2, lmax + 1)])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos = float(np.dot(v1, v2) / (n1 * n2))
            else:
                cos = 0.0
            cosines[f"{l1}_{l2}"] = cos

    return cosines


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Mask-Variation D3 Robustness Test")
    parser.add_argument('--mapdir', required=True, help='Directory with Planck CMB maps')
    parser.add_argument('--maskdir', required=True, help='Directory with downloaded masks')
    parser.add_argument('--galmaskfile', default=None, help='GAL mask file (HFI_Mask_GalPlane-apo0)')
    parser.add_argument('--outdir', required=True, help='Output directory')
    parser.add_argument('--nmc', type=int, default=5000)
    parser.add_argument('--lmax', type=int, default=15)
    parser.add_argument('--nside', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()
    nside = args.nside
    lmax = args.lmax
    nmc = args.nmc

    print("=" * 72)
    print("MASK-VARIATION D3 ROBUSTNESS TEST")
    print(f"  NSIDE={nside}, lmax={lmax}, N_MC={nmc}")
    print(f"  D3 axis: ({D3_AXIS_L}, {D3_AXIS_B})")
    print("=" * 72)

    # ================================================================
    # STEP 1: CONSTRUCT ALL MASKS
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 1: CONSTRUCTING MASKS")
    print(f"{'='*72}")

    masks = {}  # label -> (mask_array_at_nside, f_sky)

    # 1a. No mask (baseline)
    masks['no_mask'] = (None, 1.0)
    print(f"  no_mask: f_sky = 1.000")

    # 1b. Pipeline confidence masks
    cmb_map_files = {
        'SMICA': 'COM_CMB_IQU-smica_2048_R3.00_full.fits',
        'NILC': 'COM_CMB_IQU-nilc_2048_R3.00_full.fits',
        'Commander': 'COM_CMB_IQU-commander_2048_R3.00_full.fits',
        'SEVEM': 'COM_CMB_IQU-sevem_2048_R3.00_full.fits',
    }

    for pipe_name, fname in cmb_map_files.items():
        fpath = os.path.join(args.mapdir, fname)
        if os.path.exists(fpath):
            m = load_confidence_mask(fpath, nside)
            if m is not None:
                f = float(np.mean(m > 0.5))
                label = f"{pipe_name.lower()}_conf"
                masks[label] = (m, f)
                print(f"  {label}: f_sky = {f:.3f}")

    # 1c. UT78 common mask
    ut78_path = os.path.join(args.maskdir, "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")
    if os.path.exists(ut78_path):
        m_ut78 = load_planck_mask(ut78_path, field=0, nside_out=nside)
        f = float(np.mean(m_ut78 > 0.5))
        masks['UT78'] = (m_ut78, f)
        print(f"  UT78: f_sky = {f:.3f}")

    # 1d. GAL series masks
    gal_file = args.galmaskfile
    if gal_file and os.path.exists(gal_file):
        gal_fields = {'GAL020': 0, 'GAL040': 1, 'GAL060': 2, 'GAL070': 3}
        for label, field in gal_fields.items():
            try:
                m = hp.read_map(gal_file, field=field, nest=True)
                m = hp.reorder(m, n2r=True).astype(np.float64)
                m = hp.ud_grade(m, nside)
                m = (m > 0.5).astype(np.float64)
                f = float(np.mean(m > 0.5))
                masks[label] = (m, f)
                print(f"  {label}: f_sky = {f:.3f}")
            except Exception as e:
                print(f"  WARNING: could not load {label}: {e}")

    # 1e. Simple latitude cuts
    for bcut in [10, 20, 30, 40, 50]:
        m = make_latitude_mask(nside, bcut)
        f = float(np.mean(m > 0.5))
        masks[f'lat{bcut}'] = (m, f)
        print(f"  lat{bcut}: f_sky = {f:.3f}")

    # 1f. Hemisphere masks (for Step 6)
    for hemi in ['north', 'south']:
        for coord in ['galactic', 'ecliptic']:
            m = make_hemisphere_mask(nside, hemi, coord)
            f = float(np.mean(m > 0.5))
            label = f"{hemi}_{'gal' if coord == 'galactic' else 'ecl'}"
            masks[label] = (m, f)
            print(f"  {label}: f_sky = {f:.3f}")

    # 1g. Apodized UT78
    if 'UT78' in masks:
        for fwhm in [5, 10]:
            m_apod = apodize_mask(masks['UT78'][0], fwhm)
            f = float(np.mean(m_apod > 0.01))  # Apodized doesn't have hard boundary
            masks[f'UT78_apod{fwhm}'] = (m_apod, f)
            print(f"  UT78_apod{fwhm}: effective f_sky ~ {f:.3f}")

    print(f"\n  Total masks constructed: {len(masks)}")

    # ================================================================
    # STEP 2: LOAD SMICA MAP
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 2: LOADING SMICA MAP")
    print(f"{'='*72}")

    smica_path = os.path.join(args.mapdir, 'COM_CMB_IQU-smica_2048_R3.00_full.fits')
    smica_raw = hp.read_map(smica_path, field=0)
    smica_map = hp.ud_grade(smica_raw, nside)
    print(f"  SMICA loaded: NSIDE={nside}, npix={len(smica_map)}")

    # ================================================================
    # STEP 3: D3 AT EACH MASK (SMICA)
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 3: D3 AT EACH MASK LEVEL (SMICA)")
    print(f"{'='*72}")

    # Define the mask suite in order of increasing aggressiveness
    mask_suite = [
        'no_mask', 'smica_conf', 'commander_conf',
        'UT78', 'GAL020', 'lat10', 'lat20',
        'GAL040', 'lat30', 'GAL060', 'lat40', 'GAL070', 'lat50',
    ]
    # Filter to available masks
    mask_suite = [m for m in mask_suite if m in masks]

    smica_results = {}
    stability_data = []

    for mask_label in mask_suite:
        mask_arr, f_sky = masks[mask_label]
        print(f"\n--- {mask_label} (f_sky={f_sky:.3f}) ---")

        result, mc_data, alm_gal = run_d3_masked(
            smica_map, mask_arr, lmax, nmc, nside,
            label=f"SMICA_{mask_label}",
            rng_seed=42 + hash(mask_label) % 10000
        )
        result['experiment'] = 'Planck_PR3'
        result['map'] = 'SMICA'
        result['mask'] = mask_label

        # Save individual result
        outpath = os.path.join(args.outdir, f"d3_smica_{mask_label}.json")
        with open(outpath, 'w') as f:
            json.dump(result, f, indent=2)

        smica_results[mask_label] = result

        stability_data.append({
            'label': mask_label,
            'f_sky': f_sky,
            'fisher_pte': result['fisher_pte_low_l'],
            'f_A2_l3': result['per_l']['3']['f_A2'],
            'f_A2_l7': result['per_l']['7']['f_A2'],
            'f_A2_l9': result['per_l']['9']['f_A2'],
            'delta_A2_l3': result['per_l']['3']['delta_A2'],
            'delta_A2_l7': result['per_l']['7']['delta_A2'],
            'delta_A2_l9': result['per_l']['9']['delta_A2'],
            'parity_ratio': result['parity_gate']['parity_ratio'],
            'inter_irrep_r': result['inter_irrep_r'],
            'driving': result['driving_multipoles'],
            'f_m3_l3': result['f_m3_l3'],
        })

    # ================================================================
    # STEP 4: APODIZATION TEST
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 4: APODIZATION TEST")
    print(f"{'='*72}")

    apod_results = {}
    for apod_label in ['UT78', 'UT78_apod5', 'UT78_apod10']:
        if apod_label in masks:
            mask_arr, f_sky = masks[apod_label]
            if apod_label not in smica_results:
                result, _, _ = run_d3_masked(
                    smica_map, mask_arr, lmax, nmc, nside,
                    label=f"SMICA_{apod_label}",
                    rng_seed=42 + hash(apod_label) % 10000
                )
                result['experiment'] = 'Planck_PR3'
                result['map'] = 'SMICA'
                result['mask'] = apod_label
                result['apodized'] = 'apod' in apod_label

                outpath = os.path.join(args.outdir, f"d3_smica_{apod_label}.json")
                with open(outpath, 'w') as f:
                    json.dump(result, f, indent=2)
                apod_results[apod_label] = result
            else:
                apod_results[apod_label] = smica_results[apod_label]

    # ================================================================
    # STEP 5: AXIS RE-OPTIMIZATION
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 5: AXIS RE-OPTIMIZATION AT KEY MASKS")
    print(f"{'='*72}")

    axis_stability = []
    axis_masks_to_test = ['no_mask', 'UT78', 'GAL040', 'GAL060']
    axis_masks_to_test = [m for m in axis_masks_to_test if m in masks]

    for mask_label in axis_masks_to_test:
        mask_arr, f_sky = masks[mask_label]
        print(f"\n  Axis search: {mask_label} (f_sky={f_sky:.3f})...")

        # Apply mask and get alm
        if mask_arr is not None:
            masked = smica_map.copy() * mask_arr
            masked[mask_arr < 0.5] = 0.0
        else:
            masked = smica_map.copy()

        alm_gal = hp.map2alm(masked, lmax=lmax)
        opt_l, opt_b, opt_fA2, sep_deg = axis_search_masked(alm_gal, lmax, nside_grid=16)
        print(f"    Optimal axis: ({opt_l:.1f}, {opt_b:.1f}), f_A2(l=3)={opt_fA2:.3f}, "
              f"sep from D3={sep_deg:.1f} deg")

        axis_stability.append({
            'label': mask_label,
            'f_sky': f_sky,
            'axis_l': float(opt_l),
            'axis_b': float(opt_b),
            'f_A2_l3_at_opt': float(opt_fA2),
            'shift_deg': float(sep_deg),
        })

    # Save axis stability
    axis_out = {'test': 'axis_stability', 'map': 'SMICA', 'axes': axis_stability}
    with open(os.path.join(args.outdir, "axis_stability.json"), 'w') as f:
        json.dump(axis_out, f, indent=2)

    # ================================================================
    # STEP 6: CROSS-MAP AT UT78 AND GAL040
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 6: CROSS-MAP REPLICATION UNDER MASKING")
    print(f"{'='*72}")

    crossmap_data = {'test': 'crossmap_masked', 'masks_tested': [], 'per_mask': {}}

    for cross_mask_label in ['no_mask', 'UT78', 'GAL040']:
        if cross_mask_label not in masks:
            continue

        mask_arr, f_sky = masks[cross_mask_label]
        print(f"\n--- Cross-map at {cross_mask_label} (f_sky={f_sky:.3f}) ---")

        # Generate MC for this mask once
        rng = np.random.default_rng(42 + hash(cross_mask_label) % 10000)
        mc_data = masked_mc_d3(mask_arr, nside, lmax, nmc, rng)

        map_results = {}
        for pipe_name, fname in cmb_map_files.items():
            fpath = os.path.join(args.mapdir, fname)
            if not os.path.exists(fpath):
                continue

            raw = hp.read_map(fpath, field=0)
            sky = hp.ud_grade(raw, nside)

            result, _, _ = run_d3_masked(
                sky, mask_arr, lmax, nmc, nside,
                label=f"{pipe_name}_{cross_mask_label}",
                rng_seed=42, mc_data=mc_data
            )
            result['experiment'] = 'Planck_PR3'
            result['map'] = pipe_name
            result['mask'] = cross_mask_label

            # Save
            outpath = os.path.join(args.outdir,
                                    f"d3_{pipe_name.lower()}_{cross_mask_label}.json")
            with open(outpath, 'w') as f:
                json.dump(result, f, indent=2)

            map_results[pipe_name] = result

        # Pairwise cosines
        cosines = pairwise_cosines(map_results, lmax)
        cos_vals = list(cosines.values())
        entry = {
            'pairwise_cosine': cosines,
            'min_cosine': float(min(cos_vals)) if cos_vals else 0,
            'mean_cosine': float(np.mean(cos_vals)) if cos_vals else 0,
            'fisher_ptes': {k: v['fisher_pte_low_l'] for k, v in map_results.items()},
        }
        crossmap_data['masks_tested'].append(cross_mask_label)
        crossmap_data['per_mask'][cross_mask_label] = entry

        print(f"  Pairwise cosines: min={entry['min_cosine']:.3f}, mean={entry['mean_cosine']:.3f}")

    # Verdict
    if len(crossmap_data['per_mask']) >= 2:
        nomask_mean = crossmap_data['per_mask'].get('no_mask', {}).get('mean_cosine', 0)
        masked_means = [v['mean_cosine'] for k, v in crossmap_data['per_mask'].items()
                       if k != 'no_mask']
        if masked_means and min(masked_means) > 0.8:
            crossmap_data['verdict'] = 'cross-map replication survives masking'
        elif masked_means and min(masked_means) > 0.5:
            crossmap_data['verdict'] = 'cross-map replication partially survives'
        else:
            crossmap_data['verdict'] = 'cross-map replication degrades under masking'

    with open(os.path.join(args.outdir, "crossmap_masked.json"), 'w') as f:
        json.dump(crossmap_data, f, indent=2)

    # ================================================================
    # STEP 7: HEMISPHERE SPLITS
    # ================================================================
    print(f"\n{'='*72}")
    print("STEP 7: HEMISPHERE SPLITS")
    print(f"{'='*72}")

    for hemi_label in ['north_gal', 'south_gal', 'north_ecl', 'south_ecl']:
        if hemi_label not in masks:
            continue
        mask_arr, f_sky = masks[hemi_label]
        print(f"\n--- {hemi_label} (f_sky={f_sky:.3f}) ---")

        result, _, _ = run_d3_masked(
            smica_map, mask_arr, lmax, nmc, nside,
            label=f"SMICA_{hemi_label}",
            rng_seed=42 + hash(hemi_label) % 10000
        )
        result['experiment'] = 'Planck_PR3'
        result['map'] = 'SMICA'
        result['mask'] = hemi_label

        outpath = os.path.join(args.outdir, f"d3_smica_{hemi_label}.json")
        with open(outpath, 'w') as f:
            json.dump(result, f, indent=2)

    # ================================================================
    # SAVE SUMMARY
    # ================================================================
    print(f"\n{'='*72}")
    print("SAVING MASK STABILITY SUMMARY")
    print(f"{'='*72}")

    # Verdict
    if stability_data:
        # Check: does Fisher PTE stay < 0.05 through UT78?
        ut78_data = next((d for d in stability_data if d['label'] == 'UT78'), None)
        gal040_data = next((d for d in stability_data if d['label'] == 'GAL040'), None)

        if ut78_data and ut78_data['fisher_pte'] < 0.05:
            verdict = 'stable'
        elif ut78_data and ut78_data['fisher_pte'] < 0.15:
            verdict = 'partially stable'
        else:
            verdict = 'check mask dependence'

        # Check axis stability
        max_shift = max(a['shift_deg'] for a in axis_stability) if axis_stability else 0

        stability_summary = {
            'test': 'mask_stability',
            'map': 'SMICA',
            'n_mc': nmc,
            'masks': stability_data,
            'axis_shifts': axis_stability,
            'max_axis_shift_deg': float(max_shift),
            'verdict': verdict,
        }
    else:
        stability_summary = {'test': 'mask_stability', 'verdict': 'no data'}

    with open(os.path.join(args.outdir, "mask_stability.json"), 'w') as f:
        json.dump(stability_summary, f, indent=2)
    print(f"  Saved: mask_stability.json")

    # ================================================================
    # FINAL REPORT
    # ================================================================
    print(f"\n{'='*72}")
    print("FINAL REPORT: SIGNAL PERSISTENCE")
    print(f"{'='*72}")

    print(f"\n  {'Mask':<20} {'f_sky':>6} {'Fisher':>10} {'dA2(l=3)':>10} "
          f"{'dA2(l=7)':>10} {'dA2(l=9)':>10} {'Parity':>8} {'Driving':>15}")
    print(f"  {'-'*95}")
    for d in stability_data:
        driving = ','.join(str(x) for x in d['driving'])
        print(f"  {d['label']:<20} {d['f_sky']:>6.3f} {d['fisher_pte']:>10.4f} "
              f"{d['delta_A2_l3']:>+10.4f} {d['delta_A2_l7']:>+10.4f} "
              f"{d['delta_A2_l9']:>+10.4f} {d['parity_ratio']:>+8.3f} {driving:>15}")

    print(f"\n  Axis stability:")
    for a in axis_stability:
        print(f"    {a['label']:<15}: ({a['axis_l']:.1f}, {a['axis_b']:.1f}), "
              f"shift={a['shift_deg']:.1f} deg")

    if 'UT78' in apod_results or len(apod_results) > 0:
        print(f"\n  Apodization comparison:")
        for label, r in apod_results.items():
            print(f"    {label}: Fisher={r['fisher_pte_low_l']:.4f}, "
                  f"parity={r['parity_gate']['parity_ratio']:+.3f}")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n{'='*72}")
    print("DONE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
