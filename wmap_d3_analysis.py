#!/usr/bin/env python3
"""
WMAP 9-Year D3 Replication Pipeline
====================================
Cross-experiment test: replicate the entire D3 = Weyl(SU(3)) analysis
on WMAP 9-year data (ILC, V-band, W-band) at both the Planck axis and
the WMAP-optimized axis.

Usage:
  python wmap_d3_analysis.py --datadir wmap_d3/data/ --outdir wmap_d3/results/ \
    --planck-ref extended_results/d3_extended_smica_lmax150.json \
    --witness-ref extended_results/witness_transfer_analysis.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import healpy as hp
from scipy.stats import chi2 as chi2_dist, pearsonr

# =====================================================================
# D3 AXIS (FIXED from Planck CMB analysis)
# =====================================================================
D3_AXIS_L = 50.3
D3_AXIS_B = -64.9
D3_ANTI_L = 230.3
D3_ANTI_B = 64.9

# =====================================================================
# D3 DECOMPOSITION (verbatim from d3_extended_analysis.py)
# =====================================================================
def rotate_to_d3_frame(alm, lmax, axis_l=D3_ANTI_L, axis_b=D3_ANTI_B):
    """Rotate alm so the given axis (default D3 antipode) is at the pole."""
    theta = np.radians(90.0 - axis_b)
    phi = np.radians(axis_l)
    alm_rot = alm.copy()
    hp.rotate_alm(alm_rot, -phi, -theta, 0, lmax=lmax)
    return alm_rot


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


def fisher_combine(ptes, min_l=2):
    """Fisher combined PTE from per-l PTEs."""
    valid = [p for l, p in ptes.items() if l >= min_l and 0 < p < 1]
    if not valid:
        return 1.0
    S = -2 * np.sum(np.log(valid))
    return float(chi2_dist.sf(S, 2 * len(valid)))


# =====================================================================
# MAP LOADING
# =====================================================================
def load_wmap_map(fits_path, field=0, is_iqumap=False):
    """Load a WMAP FITS map, handling NESTED ordering.

    WMAP maps use NESTED ordering. We convert to RING for healpy.
    For IQU maps, field=0 gives temperature.
    """
    print(f"  Loading: {Path(fits_path).name}")
    m = hp.read_map(fits_path, field=field, nest=True, verbose=False)
    # Convert NESTED -> RING
    nside = hp.npix2nside(len(m))
    m_ring = hp.reorder(m, n2r=True)
    print(f"    NSIDE={nside}, npix={len(m)}")
    return m_ring, nside


def load_wmap_mask(fits_path):
    """Load WMAP temperature analysis mask."""
    print(f"  Loading mask: {Path(fits_path).name}")
    # WMAP masks may be in different HDU structures
    try:
        m = hp.read_map(fits_path, field=0, nest=True, verbose=False)
    except Exception:
        m = hp.read_map(fits_path, field=0, verbose=False)
        nside = hp.npix2nside(len(m))
        f_sky = np.mean(m > 0.5)
        print(f"    NSIDE={nside}, f_sky={f_sky:.1%} (assumed RING)")
        return m, nside
    nside = hp.npix2nside(len(m))
    m_ring = hp.reorder(m, n2r=True)
    f_sky = np.mean(m_ring > 0.5)
    print(f"    NSIDE={nside}, f_sky={f_sky:.1%}")
    return m_ring, nside


def prepare_map(raw_map, mask, target_nside, lmax):
    """Downgrade map and mask, apply mask, extract alm."""
    # Downgrade
    if hp.npix2nside(len(raw_map)) != target_nside:
        m_down = hp.ud_grade(raw_map, target_nside)
    else:
        m_down = raw_map.copy()

    if hp.npix2nside(len(mask)) != target_nside:
        mask_down = hp.ud_grade(mask, target_nside)
        mask_down = (mask_down > 0.5).astype(float)
    else:
        mask_down = (mask > 0.5).astype(float)

    f_sky = np.mean(mask_down > 0.5)

    # Apply mask
    masked_map = m_down.copy()
    masked_map[mask_down < 0.5] = 0.0

    # Extract alm
    alm = hp.map2alm(masked_map, lmax=lmax)
    return alm, f_sky, masked_map


# =====================================================================
# STEP 3: BLIND AXIS SEARCH
# =====================================================================
def blind_axis_search(alm, lmax, search_nside=32, l_target=3):
    """Find the axis maximizing f_A2(l=l_target) by scanning a HEALPix grid."""
    npix = hp.nside2npix(search_nside)
    test_theta, test_phi = hp.pix2ang(search_nside, np.arange(npix))

    print(f"  Blind axis search: {npix} directions, optimizing f_A2(l={l_target})")
    t0 = time.time()

    best_fA2 = -1
    best_idx = 0
    all_fA2 = np.zeros(npix)

    for i in range(npix):
        alm_rot = alm.copy()
        hp.rotate_alm(alm_rot, -test_phi[i], -test_theta[i], 0, lmax=lmax)
        _, fA2, _ = d3_fractions(alm_rot, lmax, l_target)
        all_fA2[i] = fA2
        if fA2 > best_fA2:
            best_fA2 = fA2
            best_idx = i

    # Convert best pixel to Galactic (l, b)
    best_theta = test_theta[best_idx]
    best_phi = test_phi[best_idx]
    best_b = 90.0 - np.degrees(best_theta)
    best_l = np.degrees(best_phi)

    # Also compute the antipode (the actual D3 axis direction)
    axis_l = (best_l + 180.0) % 360.0
    axis_b = -best_b

    # Separation from Planck D3 axis
    # Using great-circle distance
    cos_sep = (np.sin(np.radians(axis_b)) * np.sin(np.radians(D3_AXIS_B)) +
               np.cos(np.radians(axis_b)) * np.cos(np.radians(D3_AXIS_B)) *
               np.cos(np.radians(axis_l - D3_AXIS_L)))
    sep_deg = np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))
    # Also check antipodal separation
    cos_sep_anti = (np.sin(np.radians(best_b)) * np.sin(np.radians(D3_AXIS_B)) +
                    np.cos(np.radians(best_b)) * np.cos(np.radians(D3_AXIS_B)) *
                    np.cos(np.radians(best_l - D3_AXIS_L)))
    sep_anti = np.degrees(np.arccos(np.clip(cos_sep_anti, -1, 1)))
    # Take the minimum (axis direction is ambiguous up to antipode)
    sep_deg = min(sep_deg, sep_anti)

    # f_A2 at Planck axis
    planck_theta = np.radians(90.0 - D3_ANTI_B)
    planck_phi = np.radians(D3_ANTI_L)
    alm_planck = alm.copy()
    hp.rotate_alm(alm_planck, -planck_phi, -planck_theta, 0, lmax=lmax)
    _, fA2_planck, _ = d3_fractions(alm_planck, lmax, l_target)

    # Top 10 axes
    top10_idx = np.argsort(all_fA2)[::-1][:10]
    top10 = []
    for idx in top10_idx:
        t_b = 90.0 - np.degrees(test_theta[idx])
        t_l = np.degrees(test_phi[idx])
        # Report as axis direction (not pole)
        t_axis_l = (t_l + 180.0) % 360.0
        t_axis_b = -t_b
        top10.append({
            "l": float(t_axis_l), "b": float(t_axis_b),
            "f_A2_l3": float(all_fA2[idx]),
        })

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    print(f"    WMAP optimal axis: (l,b) = ({axis_l:.1f}, {axis_b:.1f})")
    print(f"    f_A2(l={l_target}) at WMAP axis: {best_fA2:.4f}")
    print(f"    f_A2(l={l_target}) at Planck axis: {fA2_planck:.4f}")
    print(f"    Separation from Planck axis: {sep_deg:.1f} deg")

    result = {
        "experiment": "WMAP9",
        "map": "ILC",
        "search_nside": search_nside,
        "n_axes_tested": npix,
        "l_target": l_target,
        "wmap_optimal_axis": {"l": float(axis_l), "b": float(axis_b)},
        "wmap_optimal_pole": {"l": float(best_l), "b": float(best_b)},
        "planck_axis": {"l": D3_AXIS_L, "b": D3_AXIS_B},
        "separation_deg": float(sep_deg),
        "f_A2_at_wmap_axis": float(best_fA2),
        "f_A2_at_planck_axis": float(fA2_planck),
        "top_10_axes": top10,
    }
    return result, axis_l, axis_b


# =====================================================================
# CORE D3 ANALYSIS (Steps 4-5)
# =====================================================================
def run_d3_analysis(alm, lmax, n_mc=10000, n_axes=1000,
                    mask=None, nside=256, f_sky=0.78,
                    axis_l=D3_ANTI_L, axis_b=D3_ANTI_B,
                    label="", cl_fid=None, verbose=True):
    """Run complete D3 analysis with Gaussian MC and axis scramble."""
    if verbose:
        print(f"\n  --- D3 Analysis: {label} ---")
        print(f"    NSIDE={nside}, lmax={lmax}, f_sky={f_sky:.1%}")
        print(f"    D3 axis pole: (l,b)=({axis_l:.1f}, {axis_b:.1f})")

    # Rotate to D3 frame
    alm_d3 = rotate_to_d3_frame(alm, lmax, axis_l=axis_l, axis_b=axis_b)

    # Per-l decomposition
    per_l = {}
    for l in range(1, lmax + 1):
        fA1, fA2, fE = d3_fractions(alm_d3, lmax, l)
        _, _, _, efA1, efA2, efE = d3_irrep_dimensions(l)
        per_l[l] = {
            "f_A1": float(fA1), "f_A2": float(fA2), "f_E": float(fE),
            "ef_A1": float(efA1), "ef_A2": float(efA2), "ef_E": float(efE),
            "delta_A2": float(fA2 - efA2),
        }

    if verbose:
        print(f"    {'l':>3s} {'f_A2':>8s} {'ef_A2':>8s} {'dA2':>8s}")
        for l in range(2, min(lmax + 1, 16)):
            d = per_l[l]
            flag = " ***" if d["delta_A2"] > 0.1 else ""
            print(f"    {l:3d} {d['f_A2']:8.4f} {d['ef_A2']:8.4f} "
                  f"{d['delta_A2']:+8.4f}{flag}")

    # --- Gaussian MC ---
    if n_mc > 0 and cl_fid is not None:
        rng = np.random.default_rng(42)
        mc_fA2 = {l: [] for l in range(1, lmax + 1)}
        t0 = time.time()

        # Prepare mask for MC
        if mask is not None:
            npix = hp.nside2npix(nside)
            if len(mask) != npix:
                mask_mc = hp.ud_grade(mask, nside)
                mask_mc = (mask_mc > 0.5).astype(float)
            else:
                mask_mc = (mask > 0.5).astype(float)
        else:
            mask_mc = None

        for i in range(n_mc):
            if verbose and (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"      MC {i+1}/{n_mc} ({rate:.0f}/s)")

            # Generate isotropic Gaussian sky from C_l
            sim_alm = hp.synalm(cl_fid, lmax=lmax, new=True)

            # Apply mask via map space if mask provided
            if mask_mc is not None:
                sim_map = hp.alm2map(sim_alm, nside, verbose=False)
                sim_map[mask_mc < 0.5] = 0.0
                sim_alm = hp.map2alm(sim_map, lmax=lmax)

            # Rotate to D3 frame
            sim_d3 = rotate_to_d3_frame(sim_alm, lmax, axis_l=axis_l, axis_b=axis_b)
            for l in range(1, lmax + 1):
                _, f2, _ = d3_fractions(sim_d3, lmax, l)
                mc_fA2[l].append(f2)

        for l in mc_fA2:
            mc_fA2[l] = np.array(mc_fA2[l])

        if verbose:
            print(f"    MC done: {n_mc} sims in {time.time()-t0:.1f}s")

        for l in range(1, lmax + 1):
            mc = mc_fA2[l]
            obs = per_l[l]["f_A2"]
            mc_mean = float(np.mean(mc))
            mc_std = float(np.std(mc))
            pte = float(np.mean(mc >= obs))
            z = (obs - mc_mean) / mc_std if mc_std > 0 else 0.0
            per_l[l].update({
                "pte_A2": pte, "z_A2": float(z),
                "mc_mean_A2": mc_mean, "mc_std_A2": mc_std,
            })
    else:
        # No MC — just set None
        for l in range(1, lmax + 1):
            per_l[l].update({"pte_A2": None, "z_A2": None})

    if verbose and n_mc > 0:
        print(f"    {'l':>3s} {'f_A2':>8s} {'PTE':>8s} {'z':>8s}")
        for l in range(2, min(lmax + 1, 16)):
            d = per_l[l]
            pte_val = d.get("pte_A2")
            pte_val = pte_val if pte_val is not None else 1.0
            z_val = d.get("z_A2", 0) or 0
            flag = " ***" if pte_val < 0.05 else ""
            print(f"    {l:3d} {d['f_A2']:8.4f} {pte_val:8.4f} {z_val:8.2f}{flag}")

    # --- Fisher PTEs ---
    fisher_ptes_low = {}
    fisher_ptes_all = {}
    for l in range(2, lmax + 1):
        pte = per_l[l].get("pte_A2")
        if pte is not None:
            fisher_ptes_all[l] = pte
            if l <= 15:
                fisher_ptes_low[l] = pte

    fisher_low = fisher_combine(fisher_ptes_low)
    fisher_all = fisher_combine(fisher_ptes_all)

    # --- Axis scramble ---
    axis_ptes = {}
    if n_axes > 0:
        npix_grid = hp.nside2npix(16)
        test_theta, test_phi = hp.pix2ang(16, np.arange(npix_grid))
        if npix_grid > n_axes:
            rng_ax = np.random.default_rng(123)
            idx_sel = rng_ax.choice(npix_grid, size=n_axes, replace=False)
            test_theta = test_theta[idx_sel]
            test_phi = test_phi[idx_sel]

        scramble_fA2 = {l: [] for l in range(1, lmax + 1)}
        t0 = time.time()

        for i in range(len(test_theta)):
            alm_rot = alm.copy()
            hp.rotate_alm(alm_rot, -test_phi[i], -test_theta[i], 0, lmax=lmax)
            for l in range(1, lmax + 1):
                _, f2, _ = d3_fractions(alm_rot, lmax, l)
                scramble_fA2[l].append(f2)

        for l in range(2, lmax + 1):
            sc = np.array(scramble_fA2[l])
            obs = per_l[l]["f_A2"]
            pte = float(np.mean(sc >= obs))
            per_l[l]["axis_scramble_pte"] = pte
            if l <= 15:
                axis_ptes[l] = pte

        if verbose:
            print(f"    Axis scramble: {len(test_theta)} dirs in {time.time()-t0:.1f}s")

    fisher_axis = fisher_combine(axis_ptes) if axis_ptes else 1.0

    # --- Parity gate (l=2-15) ---
    odd_ls = [l for l in range(3, min(lmax + 1, 16), 2)]
    even_ls = [l for l in range(2, min(lmax + 1, 16), 2)]
    odd_sum = sum(per_l[l]["delta_A2"] for l in odd_ls)
    even_sum = sum(per_l[l]["delta_A2"] for l in even_ls)
    odd_pos = sum(1 for l in odd_ls if per_l[l]["delta_A2"] > 0)
    even_pos = sum(1 for l in even_ls if per_l[l]["delta_A2"] > 0)
    denom = abs(odd_sum) + abs(even_sum)
    parity_ratio = (odd_sum - even_sum) / denom if denom > 0 else 0.0

    parity_gate = {
        "odd_l_values": odd_ls,
        "even_l_values": even_ls,
        "odd_l_sum_delta_A2": float(odd_sum),
        "even_l_sum_delta_A2": float(even_sum),
        "odd_l_positive_count": f"{odd_pos}/{len(odd_ls)}",
        "even_l_positive_count": f"{even_pos}/{len(even_ls)}",
        "parity_asymmetry_ratio": float(parity_ratio),
    }

    # --- E -> A2 funnel (l=2-15) ---
    dA2_list = [per_l[l]["delta_A2"] for l in range(2, min(lmax + 1, 16))]
    dE_list = [per_l[l]["f_E"] - per_l[l]["ef_E"] for l in range(2, min(lmax + 1, 16))]
    r_EA2 = float(pearsonr(dA2_list, dE_list)[0]) if len(dA2_list) >= 3 else 0.0

    # --- Cumulative transfer invariant Xi(L) ---
    xi_cumulative = []
    xi_running = 0.0
    for L in range(2, lmax + 1):
        xi_running += per_l[L]["delta_A2"] * (2 * L + 1)
        xi_cumulative.append({"L": L, "xi": float(xi_running)})

    if verbose:
        print(f"\n    Parity gate (l=2-15):")
        print(f"      Odd-l  sum dA2 = {odd_sum:+.4f} ({odd_pos}/{len(odd_ls)} positive)")
        print(f"      Even-l sum dA2 = {even_sum:+.4f} ({even_pos}/{len(even_ls)} positive)")
        print(f"      Ratio: {parity_ratio:+.4f}")
        print(f"    E->A2 funnel r: {r_EA2:.4f}")
        print(f"    Fisher PTE (l=2-15): {fisher_low:.6f}")
        print(f"    Fisher PTE (axis, l=2-15): {fisher_axis:.6f}")
        if lmax > 15:
            print(f"    Fisher PTE (all l): {fisher_all:.6f}")

    result = {
        "experiment": "WMAP9",
        "map": label,
        "nside": nside,
        "lmax": lmax,
        "f_sky": float(f_sky),
        "d3_axis": {"l": float(axis_l - 180.0) if axis_l > 180 else float(axis_l),
                     "b": float(-axis_b)},
        "d3_axis_pole": {"l": float(axis_l), "b": float(axis_b)},
        "per_l": {str(l): per_l[l] for l in range(1, lmax + 1)},
        "fisher_pte_low_l": float(fisher_low),
        "fisher_pte_all": float(fisher_all),
        "fisher_axis_scramble_pte": float(fisher_axis),
        "n_mc": n_mc,
        "n_axis_scramble": n_axes,
        "parity_gate": parity_gate,
        "inter_irrep_r": float(r_EA2),
        "xi_cumulative": xi_cumulative,
    }
    return result


# =====================================================================
# GRAMMAR COMPARISON (Step 5c, 5d)
# =====================================================================
def grammar_comparison(wmap_result, planck_ref, witness_ref, outdir):
    """Compare WMAP results to Planck SMICA grammar."""
    print("\n  === GRAMMAR COMPARISON: WMAP vs Planck ===")

    comparison = {
        "comparison": "WMAP_vs_Planck",
        "planck_map": "SMICA",
        "wmap_map": wmap_result["map"],
    }

    # 5c. Per-l cosine similarity of (f_A1, f_A2, f_E) vectors
    cos_sims = {}
    wmap_driving = []
    planck_driving = []
    for l in range(2, 16):
        w = wmap_result["per_l"][str(l)]
        p = planck_ref["per_l"][str(l)]
        w_vec = np.array([w["f_A1"], w["f_A2"], w["f_E"]])
        p_vec = np.array([p["f_A1"], p["f_A2"], p["f_E"]])
        norm_w = np.linalg.norm(w_vec)
        norm_p = np.linalg.norm(p_vec)
        if norm_w > 0 and norm_p > 0:
            cos_sim = float(np.dot(w_vec, p_vec) / (norm_w * norm_p))
        else:
            cos_sim = 0.0
        cos_sims[str(l)] = cos_sim

        # Driving multipoles (significant A2 excess)
        if w["delta_A2"] > 0.1 and w.get("pte_A2") is not None:
            pte = w["pte_A2"]
            if pte < 0.1:
                wmap_driving.append(l)
        if p["delta_A2"] > 0.1:
            planck_driving.append(l)

    comparison["per_l_cosine_similarity"] = cos_sims

    # Overall f-vector cosine (concatenate all l=2-15)
    w_all = []
    p_all = []
    for l in range(2, 16):
        w = wmap_result["per_l"][str(l)]
        p = planck_ref["per_l"][str(l)]
        w_all.extend([w["f_A1"], w["f_A2"], w["f_E"]])
        p_all.extend([p["f_A1"], p["f_A2"], p["f_E"]])
    w_all = np.array(w_all)
    p_all = np.array(p_all)
    overall_cos = float(np.dot(w_all, p_all) / (np.linalg.norm(w_all) * np.linalg.norm(p_all)))
    comparison["overall_f_vector_cosine"] = overall_cos

    # 5c. Rank-1 alignment with Planck SVD
    if witness_ref and "SMICA" in witness_ref:
        svd = witness_ref["SMICA"]["svd"]
        v_planck = np.array(svd["v"])  # (3,) = A1, A2, E singular vector

        # Project WMAP per-l irrep fractions onto Planck v
        projections = {}
        for l in range(2, 16):
            w = wmap_result["per_l"][str(l)]
            w_vec = np.array([w["f_A1"], w["f_A2"], w["f_E"]])
            proj = float(np.dot(w_vec, v_planck) / np.linalg.norm(v_planck))
            projections[str(l)] = proj

        comparison["planck_svd_v"] = list(v_planck)
        comparison["wmap_projection_onto_planck_v"] = projections

    # Parity gate match
    w_pg = wmap_result["parity_gate"]
    w_odd_dom = w_pg["odd_l_sum_delta_A2"] > w_pg["even_l_sum_delta_A2"]
    comparison["parity_gate_match"] = bool(w_odd_dom)
    comparison["wmap_parity_ratio"] = w_pg["parity_asymmetry_ratio"]

    # Xi correlation (comparing running sums)
    w_xi = {x["L"]: x["xi"] for x in wmap_result["xi_cumulative"]}
    p_xi_vals = []
    w_xi_vals = []
    for l in range(2, min(16, max(w_xi.keys()) + 1)):
        if str(l) in planck_ref["per_l"]:
            # Reconstruct Planck xi
            p_running = 0.0
            for ll in range(2, l + 1):
                p_running += planck_ref["per_l"][str(ll)]["delta_A2"] * (2 * ll + 1)
            p_xi_vals.append(p_running)
            w_xi_vals.append(w_xi.get(l, 0.0))

    if len(p_xi_vals) >= 3:
        xi_corr = float(pearsonr(w_xi_vals, p_xi_vals)[0])
    else:
        xi_corr = 0.0
    comparison["xi_correlation_l2_15"] = xi_corr

    comparison["driving_multipoles_wmap"] = wmap_driving
    comparison["driving_multipoles_planck"] = [3, 7, 9]

    # Inter-irrep correlation
    comparison["wmap_inter_irrep_r"] = wmap_result["inter_irrep_r"]
    comparison["planck_inter_irrep_r"] = planck_ref.get("combined", {}).get(
        "E_A2_correlation", -0.805)

    # Verdict
    fisher_low = wmap_result["fisher_pte_low_l"]
    axis_pte = wmap_result["fisher_axis_scramble_pte"]
    if fisher_low < 0.05 and axis_pte < 0.1:
        verdict = "REPLICATED — WMAP confirms D3 grammar at Planck axis"
    elif fisher_low < 0.1 and axis_pte < 0.2:
        verdict = "PARTIALLY REPLICATED — marginal D3 signal in WMAP"
    elif fisher_low < 0.2:
        verdict = "WEAKLY SUGGESTIVE — some D3 features present but not significant"
    else:
        verdict = "NOT REPLICATED — WMAP does not show D3 signal at this axis"

    comparison["verdict"] = verdict

    print(f"    Per-l cosine similarity (l=2-15):")
    for l in range(2, 16):
        cs = cos_sims[str(l)]
        w = wmap_result["per_l"][str(l)]
        p = planck_ref["per_l"][str(l)]
        print(f"      l={l:2d}: cos={cs:+.4f}  WMAP dA2={w['delta_A2']:+.4f}  "
              f"Planck dA2={p['delta_A2']:+.4f}")
    print(f"    Overall f-vector cosine: {overall_cos:.4f}")
    print(f"    Xi correlation (l=2-15): {xi_corr:.4f}")
    print(f"    Parity gate match: {w_odd_dom}")
    print(f"    Driving multipoles: WMAP={wmap_driving}, Planck=[3,7,9]")
    print(f"    VERDICT: {verdict}")

    return comparison


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="WMAP 9-Year D3 Replication")
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--outdir", default="wmap_d3/results")
    parser.add_argument("--planck-ref", default="extended_results/d3_extended_smica_lmax150.json")
    parser.add_argument("--witness-ref", default="extended_results/witness_transfer_analysis.json")
    parser.add_argument("--nmc", type=int, default=10000)
    parser.add_argument("--naxes", type=int, default=1000)
    parser.add_argument("--lmax-core", type=int, default=15,
                        help="Core low-l analysis limit")
    parser.add_argument("--lmax-ext", type=int, default=150,
                        help="Extended analysis limit")
    args = parser.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WMAP 9-YEAR D3 REPLICATION PIPELINE")
    print("=" * 70)

    # Load Planck reference data
    planck_ref = None
    if Path(args.planck_ref).exists():
        with open(args.planck_ref) as f:
            planck_ref = json.load(f)
        print(f"  Loaded Planck SMICA reference: {args.planck_ref}")
    else:
        print(f"  WARNING: Planck reference not found: {args.planck_ref}")

    witness_ref = None
    if Path(args.witness_ref).exists():
        with open(args.witness_ref) as f:
            witness_ref = json.load(f)
        print(f"  Loaded witness transfer reference: {args.witness_ref}")

    # ===================================================================
    # STEP 1-2: LOAD AND PREPARE MAPS
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 1-2: LOAD AND PREPARE MAPS")
    print(f"{'='*70}")

    # ILC map
    ilc_path = datadir / "wmap_ilc_9yr_v5.fits"
    if not ilc_path.exists():
        print(f"  ERROR: ILC map not found: {ilc_path}")
        sys.exit(1)
    ilc_map, ilc_nside = load_wmap_map(ilc_path)

    # KQ85 mask
    mask_path = datadir / "wmap_temperature_kq85_analysis_mask_r9_9yr_v5.fits"
    if not mask_path.exists():
        print(f"  ERROR: Mask not found: {mask_path}")
        sys.exit(1)
    kq85_mask, mask_nside = load_wmap_mask(mask_path)

    # V-band
    vband_path = datadir / "wmap_band_iqumap_r9_9yr_V_v5.fits"
    vband_map = None
    if vband_path.exists():
        vband_map, vband_nside = load_wmap_map(vband_path, field=0, is_iqumap=True)

    # W-band
    wband_path = datadir / "wmap_band_iqumap_r9_9yr_W_v5.fits"
    wband_map = None
    if wband_path.exists():
        wband_map, wband_nside = load_wmap_map(wband_path, field=0, is_iqumap=True)

    # Prepare fiducial C_l for MC (use WMAP best-fit or Planck C_l)
    # Generate a simple fiducial: use the actual map's pseudo-C_l as a starting point
    print(f"\n  Preparing fiducial C_l for MC...")
    # Low-l analysis: NSIDE=64, lmax=15
    alm_low, fsky_low, _ = prepare_map(ilc_map, kq85_mask, 64, args.lmax_core)
    cl_low = hp.alm2cl(alm_low, lmax=args.lmax_core)
    # Correct for sky fraction (approximate)
    cl_fid_low = cl_low / max(fsky_low, 0.1)
    print(f"    Low-l: NSIDE=64, lmax={args.lmax_core}, f_sky={fsky_low:.1%}")

    # Extended analysis: NSIDE=256, lmax=150
    alm_ext, fsky_ext, _ = prepare_map(ilc_map, kq85_mask, 256, args.lmax_ext)
    cl_ext = hp.alm2cl(alm_ext, lmax=args.lmax_ext)
    cl_fid_ext = cl_ext / max(fsky_ext, 0.1)
    print(f"    Extended: NSIDE=256, lmax={args.lmax_ext}, f_sky={fsky_ext:.1%}")

    # ===================================================================
    # STEP 3: BLIND AXIS SEARCH
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 3: BLIND AXIS SEARCH")
    print(f"{'='*70}")

    # Use low-l alm for axis search (l=3 octupole)
    axis_result, wmap_axis_l, wmap_axis_b = blind_axis_search(
        alm_low, args.lmax_core, search_nside=32, l_target=3)

    axis_path = outdir / "wmap_d3_axis_search.json"
    with open(axis_path, "w") as f:
        json.dump(axis_result, f, indent=2)
    print(f"  Saved: {axis_path}")

    # ===================================================================
    # STEP 4a: ILC AT PLANCK AXIS (PRIMARY TEST)
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 4a: ILC AT PLANCK AXIS — LOW-L (l=2-15)")
    print(f"{'='*70}")

    # Prepare mask at NSIDE=64 for MC
    mask_64 = hp.ud_grade(kq85_mask, 64)
    mask_64 = (mask_64 > 0.5).astype(float)

    ilc_planck_low = run_d3_analysis(
        alm_low, args.lmax_core,
        n_mc=args.nmc, n_axes=args.naxes,
        mask=mask_64, nside=64, f_sky=fsky_low,
        axis_l=D3_ANTI_L, axis_b=D3_ANTI_B,
        label="ILC_planck_axis_low_l",
        cl_fid=cl_fid_low)

    print(f"\n{'='*70}")
    print("STEP 4a: ILC AT PLANCK AXIS — EXTENDED (l=2-150)")
    print(f"{'='*70}")

    # Prepare mask at NSIDE=256 for MC
    mask_256 = hp.ud_grade(kq85_mask, 256)
    mask_256 = (mask_256 > 0.5).astype(float)

    # Use fewer MC for extended (slower per iteration at NSIDE=256)
    nmc_ext = min(args.nmc, 5000)
    ilc_planck_ext = run_d3_analysis(
        alm_ext, args.lmax_ext,
        n_mc=nmc_ext, n_axes=args.naxes,
        mask=mask_256, nside=256, f_sky=fsky_ext,
        axis_l=D3_ANTI_L, axis_b=D3_ANTI_B,
        label="ILC_planck_axis_extended",
        cl_fid=cl_fid_ext)

    # Merge low-l and extended into one result
    ilc_planck_result = ilc_planck_ext.copy()
    ilc_planck_result["fisher_pte_low_l"] = ilc_planck_low["fisher_pte_low_l"]
    ilc_planck_result["parity_gate"] = ilc_planck_low["parity_gate"]
    ilc_planck_result["inter_irrep_r"] = ilc_planck_low["inter_irrep_r"]
    # Also store low-l Fisher axis scramble
    ilc_planck_result["fisher_axis_scramble_pte_low_l"] = ilc_planck_low["fisher_axis_scramble_pte"]
    ilc_planck_result["mask"] = "KQ85"

    ilc_planck_path = outdir / "wmap_d3_ilc_planck_axis.json"
    with open(ilc_planck_path, "w") as f:
        json.dump(ilc_planck_result, f, indent=2)
    print(f"  Saved: {ilc_planck_path}")

    # ===================================================================
    # STEP 4b: V-BAND AND W-BAND AT PLANCK AXIS
    # ===================================================================
    vband_result = None
    if vband_map is not None:
        print(f"\n{'='*70}")
        print("STEP 4b: V-BAND AT PLANCK AXIS")
        print(f"{'='*70}")

        alm_v, fsky_v, _ = prepare_map(vband_map, kq85_mask, 64, args.lmax_core)
        cl_v = hp.alm2cl(alm_v, lmax=args.lmax_core) / max(fsky_v, 0.1)

        vband_result = run_d3_analysis(
            alm_v, args.lmax_core,
            n_mc=5000, n_axes=500,
            mask=mask_64, nside=64, f_sky=fsky_v,
            axis_l=D3_ANTI_L, axis_b=D3_ANTI_B,
            label="V-band_planck_axis",
            cl_fid=cl_v)
        vband_result["mask"] = "KQ85"

        vband_path = outdir / "wmap_d3_vband_planck_axis.json"
        with open(vband_path, "w") as f:
            json.dump(vband_result, f, indent=2)
        print(f"  Saved: {vband_path}")

    wband_result = None
    if wband_map is not None:
        print(f"\n{'='*70}")
        print("STEP 4b: W-BAND AT PLANCK AXIS")
        print(f"{'='*70}")

        alm_w, fsky_w, _ = prepare_map(wband_map, kq85_mask, 64, args.lmax_core)
        cl_w = hp.alm2cl(alm_w, lmax=args.lmax_core) / max(fsky_w, 0.1)

        wband_result = run_d3_analysis(
            alm_w, args.lmax_core,
            n_mc=5000, n_axes=500,
            mask=mask_64, nside=64, f_sky=fsky_w,
            axis_l=D3_ANTI_L, axis_b=D3_ANTI_B,
            label="W-band_planck_axis",
            cl_fid=cl_w)
        wband_result["mask"] = "KQ85"

        wband_path = outdir / "wmap_d3_wband_planck_axis.json"
        with open(wband_path, "w") as f:
            json.dump(wband_result, f, indent=2)
        print(f"  Saved: {wband_path}")

    # ===================================================================
    # STEP 6: ILC AT WMAP-OPTIMIZED AXIS
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 6: ILC AT WMAP-OPTIMIZED AXIS (BLIND TEST)")
    print(f"{'='*70}")

    # Convert axis to pole for rotation
    wmap_pole_l = (wmap_axis_l + 180.0) % 360.0
    wmap_pole_b = -wmap_axis_b

    ilc_wmap_axis = run_d3_analysis(
        alm_low, args.lmax_core,
        n_mc=args.nmc, n_axes=args.naxes,
        mask=mask_64, nside=64, f_sky=fsky_low,
        axis_l=wmap_pole_l, axis_b=wmap_pole_b,
        label="ILC_wmap_axis",
        cl_fid=cl_fid_low)
    ilc_wmap_axis["mask"] = "KQ85"
    ilc_wmap_axis["wmap_optimized_axis"] = {"l": float(wmap_axis_l), "b": float(wmap_axis_b)}

    wmap_axis_path = outdir / "wmap_d3_ilc_wmap_axis.json"
    with open(wmap_axis_path, "w") as f:
        json.dump(ilc_wmap_axis, f, indent=2)
    print(f"  Saved: {wmap_axis_path}")

    # ===================================================================
    # STEP 5: GRAMMAR COMPARISON
    # ===================================================================
    if planck_ref is not None:
        print(f"\n{'='*70}")
        print("STEP 5: GRAMMAR COMPARISON (WMAP vs Planck)")
        print(f"{'='*70}")

        # Use the low-l Planck-axis ILC result for grammar comparison
        grammar = grammar_comparison(ilc_planck_low, planck_ref, witness_ref, outdir)

        grammar_path = outdir / "wmap_d3_grammar_comparison.json"
        with open(grammar_path, "w") as f:
            json.dump(grammar, f, indent=2)
        print(f"  Saved: {grammar_path}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Axis search
    print(f"\n  AXIS SEARCH:")
    print(f"    WMAP optimal axis: (l,b) = ({wmap_axis_l:.1f}, {wmap_axis_b:.1f})")
    print(f"    Planck D3 axis:    (l,b) = ({D3_AXIS_L}, {D3_AXIS_B})")
    print(f"    Separation: {axis_result['separation_deg']:.1f} deg")
    print(f"    f_A2(l=3) at WMAP axis:   {axis_result['f_A2_at_wmap_axis']:.4f}")
    print(f"    f_A2(l=3) at Planck axis:  {axis_result['f_A2_at_planck_axis']:.4f}")

    # Per-l comparison table
    print(f"\n  PER-L COMPARISON (l=2-15 at Planck axis):")
    print(f"  {'l':>3s} {'WMAP f_A2':>10s} {'Planck':>10s} {'WMAP dA2':>10s} {'Planck':>10s} {'WMAP PTE':>10s}")
    for l in range(2, 16):
        w = ilc_planck_low["per_l"][str(l)]
        p_dA2 = planck_ref["per_l"][str(l)]["delta_A2"] if planck_ref else 0
        p_fA2 = planck_ref["per_l"][str(l)]["f_A2"] if planck_ref else 0
        pte = w.get("pte_A2")
        pte_str = f"{pte:.4f}" if pte is not None else "N/A"
        flag = " ***" if pte is not None and pte < 0.05 else ""
        print(f"  {l:3d} {w['f_A2']:10.4f} {p_fA2:10.4f} "
              f"{w['delta_A2']:+10.4f} {p_dA2:+10.4f} {pte_str:>10s}{flag}")

    # Scorecard
    print(f"\n  SCORECARD:")
    print(f"  {'Metric':<35s} {'WMAP ILC':>12s} {'Planck SMICA':>14s}")
    print(f"  {'-'*63}")
    print(f"  {'Fisher PTE (l=2-15)':<35s} "
          f"{ilc_planck_low['fisher_pte_low_l']:12.6f} "
          f"{planck_ref['combined']['fisher_low'] if planck_ref else 0:14.6f}")
    print(f"  {'Fisher PTE (axis, l=2-15)':<35s} "
          f"{ilc_planck_low['fisher_axis_scramble_pte']:12.6f} {'--':>14s}")
    print(f"  {'r(dA2, dE) l=2-15':<35s} "
          f"{ilc_planck_low['inter_irrep_r']:12.4f} "
          f"{planck_ref['combined']['E_A2_correlation'] if planck_ref else 0:14.4f}")
    print(f"  {'Parity ratio':<35s} "
          f"{ilc_planck_low['parity_gate']['parity_asymmetry_ratio']:12.4f} {'--':>14s}")

    # Frequency cross-check
    if vband_result or wband_result:
        print(f"\n  FREQUENCY CROSS-CHECK (l=2-15):")
        print(f"  {'Map':<15s} {'Fisher PTE':>12s} {'dA2(l=3)':>10s} {'dA2(l=7)':>10s} {'dA2(l=9)':>10s}")
        for name, res in [("ILC", ilc_planck_low),
                          ("V-band", vband_result),
                          ("W-band", wband_result)]:
            if res is None:
                continue
            fp = res["fisher_pte_low_l"]
            d3 = res["per_l"]["3"]["delta_A2"]
            d7 = res["per_l"]["7"]["delta_A2"]
            d9 = res["per_l"]["9"]["delta_A2"]
            print(f"  {name:<15s} {fp:12.6f} {d3:+10.4f} {d7:+10.4f} {d9:+10.4f}")

    # Planck vs WMAP axis comparison
    print(f"\n  PLANCK AXIS vs WMAP AXIS (ILC, l=2-15):")
    print(f"  {'Metric':<35s} {'Planck axis':>12s} {'WMAP axis':>12s}")
    print(f"  {'Fisher PTE (l=2-15)':<35s} "
          f"{ilc_planck_low['fisher_pte_low_l']:12.6f} "
          f"{ilc_wmap_axis['fisher_pte_low_l']:12.6f}")
    print(f"  {'dA2(l=3)':<35s} "
          f"{ilc_planck_low['per_l']['3']['delta_A2']:+12.4f} "
          f"{ilc_wmap_axis['per_l']['3']['delta_A2']:+12.4f}")
    print(f"  {'dA2(l=7)':<35s} "
          f"{ilc_planck_low['per_l']['7']['delta_A2']:+12.4f} "
          f"{ilc_wmap_axis['per_l']['7']['delta_A2']:+12.4f}")

    # Final verdict
    fp_low = ilc_planck_low["fisher_pte_low_l"]
    print(f"\n  VERDICT:")
    if fp_low < 0.05:
        print(f"    WMAP D3 grammar: REPLICATED (Fisher PTE = {fp_low:.6f})")
    elif fp_low < 0.1:
        print(f"    WMAP D3 grammar: MARGINAL (Fisher PTE = {fp_low:.6f})")
    elif fp_low < 0.2:
        print(f"    WMAP D3 grammar: WEAKLY SUGGESTIVE (Fisher PTE = {fp_low:.6f})")
    else:
        print(f"    WMAP D3 grammar: NOT REPLICATED (Fisher PTE = {fp_low:.6f})")


if __name__ == "__main__":
    main()
