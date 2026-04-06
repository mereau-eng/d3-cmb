#!/usr/bin/env python3
"""
Other Symmetry Groups -- D3 Specificity Test
==============================================
Test whether D3 = Weyl(SU(3)) is uniquely special among dihedral groups.
Compare D3, D4, D5, D6 decompositions at the D3 axis and at each group's
own optimized axis.

Usage:
  python group_comparison_analysis.py \
    --smica /path/to/COM_CMB_IQU-smica_2048_R3.00_full.fits \
    --nilc /path/to/COM_CMB_IQU-nilc_2048_R3.00_full.fits \
    --commander /path/to/COM_CMB_IQU-commander_2048_R3.00_full.fits \
    --sevem /path/to/COM_CMB_IQU-sevem_2048_R3.00_full.fits \
    --outdir other_groups/results/
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
# D3 AXIS (FIXED)
# =====================================================================
D3_AXIS_L = 50.3
D3_AXIS_B = -64.9
D3_ANTI_L = 230.3
D3_ANTI_B = 64.9

# =====================================================================
# GENERAL D_n DECOMPOSITION
# =====================================================================

def rotate_to_axis(alm, lmax, pole_l, pole_b):
    """Rotate alm to place (pole_l, pole_b) at the north pole."""
    theta = np.radians(90.0 - pole_b)
    phi = np.radians(pole_l)
    alm_rot = alm.copy()
    hp.rotate_alm(alm_rot, -phi, -theta, 0, lmax=lmax)
    return alm_rot


def dn_irrep_info(l, n):
    """Compute D_n irrep dimensions and expected fractions for multipole l.

    Returns dict: irrep_name -> {"dim": d, "ef": d/(2l+1)}
    """
    total = 2 * l + 1
    irreps = {}

    # Count m-values in each irrep bucket
    dim_A1 = 1  # m=0 always A1
    dim_A2 = 0
    dim_B1 = 0  # only for even n
    dim_B2 = 0

    # E irreps: for odd n, k=1..(n-1)/2; for even n, k=1..n/2-1
    if n % 2 == 0:
        n_E = n // 2 - 1
    else:
        n_E = (n - 1) // 2
    dim_E = {k: 0 for k in range(1, n_E + 1)}

    for m in range(1, l + 1):
        r = m % n
        if r == 0:
            dim_A1 += 1  # Re part
            dim_A2 += 1  # Im part
        elif n % 2 == 0 and r == n // 2:
            dim_B1 += 1  # Re part
            dim_B2 += 1  # Im part
        else:
            k = min(r, n - r)
            dim_E[k] += 2  # full +-m pair

    irreps["A1"] = {"dim": dim_A1, "ef": dim_A1 / total}
    irreps["A2"] = {"dim": dim_A2, "ef": dim_A2 / total}
    if n % 2 == 0:
        irreps["B1"] = {"dim": dim_B1, "ef": dim_B1 / total}
        irreps["B2"] = {"dim": dim_B2, "ef": dim_B2 / total}
    for k in range(1, n_E + 1):
        irreps[f"E{k}"] = {"dim": dim_E[k], "ef": dim_E[k] / total}

    return irreps


def dn_fractions(alm_rotated, lmax_alm, l, n):
    """Compute D_n irrep power fractions for multipole l.

    Returns dict: irrep_name -> fraction
    """
    total = 0.0
    power = {}
    power["A1"] = 0.0
    power["A2"] = 0.0
    if n % 2 == 0:
        power["B1"] = 0.0
        power["B2"] = 0.0

    if n % 2 == 0:
        n_E = n // 2 - 1
    else:
        n_E = (n - 1) // 2
    for k in range(1, n_E + 1):
        power[f"E{k}"] = 0.0

    for m in range(0, l + 1):
        idx = hp.Alm.getidx(lmax_alm, l, m)
        c = alm_rotated[idx]

        if m == 0:
            p = np.abs(c)**2
            total += p
            power["A1"] += p
        else:
            p = 2.0 * np.abs(c)**2
            total += p
            r = m % n
            if r == 0:
                power["A1"] += 2.0 * np.real(c)**2
                power["A2"] += 2.0 * np.imag(c)**2
            elif n % 2 == 0 and r == n // 2:
                power["B1"] += 2.0 * np.real(c)**2
                power["B2"] += 2.0 * np.imag(c)**2
            else:
                k = min(r, n - r)
                power[f"E{k}"] += p

    if total == 0:
        return {name: 0.0 for name in power}
    return {name: val / total for name, val in power.items()}


def d3_fractions_original(alm_d3, lmax_alm, l):
    """Original D3 decomposition for verification."""
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
    """Fisher combined PTE."""
    valid = [p for l, p in ptes.items() if l >= min_l and 0 < p < 1]
    if not valid:
        return 1.0
    S = -2 * np.sum(np.log(valid))
    return float(chi2_dist.sf(S, 2 * len(valid)))


# =====================================================================
# FULL D_n ANALYSIS
# =====================================================================
def run_dn_analysis(alm_gal, lmax, n, pole_l, pole_b,
                    n_mc=10000, n_axes=0,
                    cl_fid=None, nside=256, f_sky=1.0,
                    label="", verbose=True):
    """Run D_n decomposition at a specified axis."""
    group_name = f"D{n}"
    if verbose:
        print(f"\n  --- {group_name} Analysis: {label} ---")
        print(f"    Axis pole: ({pole_l:.1f}, {pole_b:.1f}), lmax={lmax}")

    # Rotate
    alm_rot = rotate_to_axis(alm_gal, lmax, pole_l, pole_b)

    # Per-l decomposition
    per_l = {}
    for l in range(1, lmax + 1):
        fracs = dn_fractions(alm_rot, lmax, l, n)
        irrep_info = dn_irrep_info(l, n)
        per_l[l] = {}
        for irrep_name in fracs:
            per_l[l][irrep_name] = {
                "f": float(fracs[irrep_name]),
                "ef": float(irrep_info[irrep_name]["ef"]),
                "delta": float(fracs[irrep_name] - irrep_info[irrep_name]["ef"]),
            }

    if verbose:
        print(f"    {'l':>3s} {'f_A2':>8s} {'ef_A2':>8s} {'dA2':>8s}", end="")
        if n % 2 == 0:
            print(f" {'f_B2':>8s} {'ef_B2':>8s}", end="")
        print()
        for l in range(2, min(lmax + 1, 16)):
            d = per_l[l]
            flag = " ***" if d["A2"]["delta"] > 0.1 else ""
            print(f"    {l:3d} {d['A2']['f']:8.4f} {d['A2']['ef']:8.4f} "
                  f"{d['A2']['delta']:+8.4f}", end="")
            if n % 2 == 0 and "B2" in d:
                print(f" {d['B2']['f']:8.4f} {d['B2']['ef']:8.4f}", end="")
            print(flag)

    # --- Gaussian MC for A2 PTEs ---
    if n_mc > 0 and cl_fid is not None:
        rng = np.random.default_rng(42)
        mc_fA2 = {l: [] for l in range(1, lmax + 1)}
        mc_fB2 = {l: [] for l in range(1, lmax + 1)} if n % 2 == 0 else None
        t0 = time.time()

        for i in range(n_mc):
            if verbose and (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f"      MC {i+1}/{n_mc} ({(i+1)/elapsed:.0f}/s)")

            sim_alm = hp.synalm(cl_fid, lmax=lmax, new=True)
            sim_rot = rotate_to_axis(sim_alm, lmax, pole_l, pole_b)

            for l in range(1, lmax + 1):
                fracs_sim = dn_fractions(sim_rot, lmax, l, n)
                mc_fA2[l].append(fracs_sim["A2"])
                if mc_fB2 is not None:
                    mc_fB2[l].append(fracs_sim.get("B2", 0.0))

        for l in range(1, lmax + 1):
            mc_fA2[l] = np.array(mc_fA2[l])
            if mc_fB2 is not None:
                mc_fB2[l] = np.array(mc_fB2[l])

        if verbose:
            print(f"    MC done: {n_mc} sims in {time.time()-t0:.1f}s")

        for l in range(1, lmax + 1):
            mc = mc_fA2[l]
            obs_A2 = per_l[l]["A2"]["f"]
            mc_mean = float(np.mean(mc))
            mc_std = float(np.std(mc))
            pte = float(np.mean(mc >= obs_A2))
            z = (obs_A2 - mc_mean) / mc_std if mc_std > 0 else 0.0
            per_l[l]["A2"].update({
                "pte": pte, "z": float(z),
                "mc_mean": mc_mean, "mc_std": mc_std,
            })
            if mc_fB2 is not None:
                mc_b = mc_fB2[l]
                obs_B2 = per_l[l].get("B2", {}).get("f", 0.0)
                if len(mc_b) > 0:
                    pte_b = float(np.mean(mc_b >= obs_B2))
                    z_b = (obs_B2 - float(np.mean(mc_b))) / max(float(np.std(mc_b)), 1e-12)
                    per_l[l]["B2"].update({"pte": pte_b, "z": float(z_b)})
    else:
        for l in range(1, lmax + 1):
            per_l[l]["A2"]["pte"] = None
            per_l[l]["A2"]["z"] = None

    if verbose and n_mc > 0:
        print(f"    {'l':>3s} {'f_A2':>8s} {'PTE':>8s} {'z':>8s}")
        for l in range(2, min(lmax + 1, 16)):
            d = per_l[l]["A2"]
            pte_val = d.get("pte")
            pte_val = pte_val if pte_val is not None else 1.0
            z_val = d.get("z", 0) or 0
            flag = " ***" if pte_val < 0.05 else ""
            print(f"    {l:3d} {d['f']:8.4f} {pte_val:8.4f} {z_val:8.2f}{flag}")

    # Fisher PTEs
    fisher_ptes_low = {}
    fisher_ptes_all = {}
    for l in range(2, lmax + 1):
        pte = per_l[l]["A2"].get("pte")
        if pte is not None:
            fisher_ptes_all[l] = pte
            if l <= 15:
                fisher_ptes_low[l] = pte

    fisher_low = fisher_combine(fisher_ptes_low)
    fisher_all = fisher_combine(fisher_ptes_all)

    # Total parity-odd Fisher (for even n: A2 + B2)
    total_parity_odd_pte = fisher_low
    if n % 2 == 0:
        po_ptes = {}
        for l in range(2, 16):
            pA2 = per_l[l]["A2"].get("pte")
            pB2 = per_l[l].get("B2", {}).get("pte")
            # Combine A2 and B2 PTEs for this l
            if pA2 is not None and pB2 is not None and 0 < pA2 < 1 and 0 < pB2 < 1:
                # Use minimum PTE (Bonferroni-style)
                po_ptes[l] = min(pA2, pB2) * 2  # rough correction for 2 tests
                po_ptes[l] = min(po_ptes[l], 1.0)
            elif pA2 is not None and 0 < pA2 < 1:
                po_ptes[l] = pA2
        total_parity_odd_pte = fisher_combine(po_ptes)

    # Axis scramble (optional)
    fisher_axis = 1.0
    if n_axes > 0:
        npix_grid = hp.nside2npix(16)
        test_theta, test_phi = hp.pix2ang(16, np.arange(npix_grid))
        if npix_grid > n_axes:
            rng_ax = np.random.default_rng(123)
            idx_sel = rng_ax.choice(npix_grid, size=n_axes, replace=False)
            test_theta = test_theta[idx_sel]
            test_phi = test_phi[idx_sel]

        axis_ptes_low = {}
        for i in range(len(test_theta)):
            alm_sc = alm_gal.copy()
            hp.rotate_alm(alm_sc, -test_phi[i], -test_theta[i], 0, lmax=lmax)
            for l in range(2, min(lmax + 1, 16)):
                if l not in axis_ptes_low:
                    axis_ptes_low[l] = []
                fracs_sc = dn_fractions(alm_sc, lmax, l, n)
                axis_ptes_low[l].append(fracs_sc["A2"])

        for l in axis_ptes_low:
            sc = np.array(axis_ptes_low[l])
            obs = per_l[l]["A2"]["f"]
            pte = float(np.mean(sc >= obs))
            per_l[l]["A2"]["axis_pte"] = pte
            axis_ptes_low[l] = pte

        fisher_axis = fisher_combine(
            {l: v for l, v in axis_ptes_low.items() if isinstance(v, float)})
        if verbose:
            print(f"    Axis scramble Fisher PTE: {fisher_axis:.6f}")

    # Parity gate (l=2-15)
    odd_ls = [l for l in range(3, min(lmax + 1, 16), 2)]
    even_ls = [l for l in range(2, min(lmax + 1, 16), 2)]
    odd_sum = sum(per_l[l]["A2"]["delta"] for l in odd_ls)
    even_sum = sum(per_l[l]["A2"]["delta"] for l in even_ls)
    odd_pos = sum(1 for l in odd_ls if per_l[l]["A2"]["delta"] > 0)
    even_pos = sum(1 for l in even_ls if per_l[l]["A2"]["delta"] > 0)
    denom = abs(odd_sum) + abs(even_sum)
    parity_ratio = (odd_sum - even_sum) / denom if denom > 0 else 0.0

    parity_gate = {
        "odd_l_sum_delta_A2": float(odd_sum),
        "even_l_sum_delta_A2": float(even_sum),
        "odd_l_positive_count": f"{odd_pos}/{len(odd_ls)}",
        "even_l_positive_count": f"{even_pos}/{len(even_ls)}",
        "parity_asymmetry_ratio": float(parity_ratio),
    }

    # E -> A2 funnel (use total E power vs A2)
    dA2_list = []
    dE_list = []
    for l in range(2, min(lmax + 1, 16)):
        dA2_list.append(per_l[l]["A2"]["delta"])
        # Sum all E irreps
        total_fE = sum(per_l[l][k]["f"] for k in per_l[l] if k.startswith("E"))
        total_efE = sum(per_l[l][k]["ef"] for k in per_l[l] if k.startswith("E"))
        dE_list.append(total_fE - total_efE)
    r_EA2 = float(pearsonr(dA2_list, dE_list)[0]) if len(dA2_list) >= 3 else 0.0

    # Xi cumulative
    xi_cumulative = []
    xi_running = 0.0
    for L in range(2, lmax + 1):
        xi_running += per_l[L]["A2"]["delta"] * (2 * L + 1)
        xi_cumulative.append({"L": L, "xi": float(xi_running)})

    if verbose:
        print(f"\n    Parity gate: odd={odd_sum:+.4f} ({odd_pos}/{len(odd_ls)}+), "
              f"even={even_sum:+.4f} ({even_pos}/{len(even_ls)}+), ratio={parity_ratio:+.4f}")
        print(f"    E->A2 funnel r: {r_EA2:.4f}")
        print(f"    Fisher PTE (l=2-15): {fisher_low:.6f}")
        print(f"    Fisher PTE (axis):   {fisher_axis:.6f}")

    result = {
        "group": group_name,
        "order": 2 * n,
        "n": n,
        "axis_pole": {"l": float(pole_l), "b": float(pole_b)},
        "axis_source": label,
        "nside": nside,
        "lmax": lmax,
        "map": "SMICA",
        "per_l": {str(l): per_l[l] for l in range(1, lmax + 1)},
        "A2_fisher_pte_low_l": float(fisher_low),
        "A2_fisher_pte_all": float(fisher_all),
        "A2_fisher_axis_scramble_pte": float(fisher_axis),
        "total_parity_odd_fisher_pte": float(total_parity_odd_pte),
        "parity_gate": parity_gate,
        "inter_irrep_r_A2_E": float(r_EA2),
        "xi_cumulative": xi_cumulative,
        "n_mc": n_mc,
    }
    return result


# =====================================================================
# AXIS SEARCH
# =====================================================================
def axis_search(alm_gal, lmax, n, search_nside=16, l_target=None):
    """Find the optimal axis for D_n by maximizing f_A2(l=l_target)."""
    group_name = f"D{n}"

    # Choose target l: first l where A2 subspace is nonzero
    if l_target is None:
        for l_try in range(2, 16):
            info = dn_irrep_info(l_try, n)
            if info["A2"]["dim"] > 0:
                l_target = l_try
                break
        if l_target is None:
            l_target = n  # fallback

    print(f"\n  Axis search for {group_name}: optimizing f_A2(l={l_target}), "
          f"grid NSIDE={search_nside}")

    npix = hp.nside2npix(search_nside)
    test_theta, test_phi = hp.pix2ang(search_nside, np.arange(npix))

    best_fA2 = -1
    best_idx = 0
    all_fA2 = np.zeros(npix)
    t0 = time.time()

    for i in range(npix):
        alm_rot = alm_gal.copy()
        hp.rotate_alm(alm_rot, -test_phi[i], -test_theta[i], 0, lmax=lmax)
        fracs = dn_fractions(alm_rot, lmax, l_target, n)
        all_fA2[i] = fracs["A2"]
        if fracs["A2"] > best_fA2:
            best_fA2 = fracs["A2"]
            best_idx = i

    best_b = 90.0 - np.degrees(test_theta[best_idx])
    best_l = np.degrees(test_phi[best_idx])
    # The "axis" direction (convention: antipode of pole)
    axis_l = (best_l + 180.0) % 360.0
    axis_b = -best_b

    # Separation from D3 axis (accounting for head/tail ambiguity)
    cos_sep1 = (np.sin(np.radians(axis_b)) * np.sin(np.radians(D3_AXIS_B)) +
                np.cos(np.radians(axis_b)) * np.cos(np.radians(D3_AXIS_B)) *
                np.cos(np.radians(axis_l - D3_AXIS_L)))
    cos_sep2 = (np.sin(np.radians(best_b)) * np.sin(np.radians(D3_AXIS_B)) +
                np.cos(np.radians(best_b)) * np.cos(np.radians(D3_AXIS_B)) *
                np.cos(np.radians(best_l - D3_AXIS_L)))
    sep1 = np.degrees(np.arccos(np.clip(cos_sep1, -1, 1)))
    sep2 = np.degrees(np.arccos(np.clip(cos_sep2, -1, 1)))
    sep_deg = min(sep1, sep2)

    # f_A2 at D3 axis
    alm_d3 = rotate_to_axis(alm_gal, lmax, D3_ANTI_L, D3_ANTI_B)
    fA2_d3 = dn_fractions(alm_d3, lmax, l_target, n)["A2"]

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    print(f"    {group_name} optimal axis: ({axis_l:.1f}, {axis_b:.1f}), "
          f"pole: ({best_l:.1f}, {best_b:.1f})")
    print(f"    f_A2(l={l_target}): optimal={best_fA2:.4f}, at D3 axis={fA2_d3:.4f}")
    print(f"    Sep from D3 axis: {sep_deg:.1f} deg")

    search_result = {
        "group": group_name, "n": n,
        "search_nside": search_nside,
        "n_axes_tested": npix,
        "l_target": l_target,
        "optimal_axis": {"l": float(axis_l), "b": float(axis_b)},
        "optimal_pole": {"l": float(best_l), "b": float(best_b)},
        "f_A2_at_optimal": float(best_fA2),
        "f_A2_at_d3_axis": float(fA2_d3),
        "separation_from_d3_deg": float(sep_deg),
    }
    return search_result, best_l, best_b


# =====================================================================
# RANK-1 FACTORIZATION (Step 4)
# =====================================================================
def rank1_test(all_maps_alm, lmax, n, pole_l, pole_b, l_range=(2, 16)):
    """Test rank-1 factorization using all 4 Planck maps.

    For D_n with k irreps, build a 4×k matrix M where M[map, irrep] = sum_l f_irrep(l).
    Actually, build per-l: for each l, the 4-vector of f_A2 values across maps.
    Then SVD to check rank-1.
    """
    n_maps = len(all_maps_alm)
    l_min, l_max = l_range

    # Collect f_A2 across maps and multipoles
    f_matrix = []  # rows = maps, cols = l values
    for alm_gal in all_maps_alm:
        alm_rot = rotate_to_axis(alm_gal, lmax, pole_l, pole_b)
        row = []
        for l in range(l_min, l_max):
            fracs = dn_fractions(alm_rot, lmax, l, n)
            row.append(fracs["A2"])
        f_matrix.append(row)

    M = np.array(f_matrix)  # shape: (4, n_l)
    if M.shape[0] < 2 or M.shape[1] < 2:
        return {"rank1_frac": 0.0, "sigma_ratio": 0.0}

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    rank1_frac = float(S[0]**2 / np.sum(S**2)) if np.sum(S**2) > 0 else 0
    sigma_ratio = float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float('inf')

    return {
        "rank1_frac": rank1_frac,
        "sigma_ratio": sigma_ratio,
        "sigma1": float(S[0]),
        "sigma2": float(S[1]) if len(S) > 1 else 0.0,
    }


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="D_n Group Specificity Test")
    parser.add_argument("--smica", required=True)
    parser.add_argument("--nilc", default=None)
    parser.add_argument("--commander", default=None)
    parser.add_argument("--sevem", default=None)
    parser.add_argument("--outdir", default="other_groups/results")
    parser.add_argument("--nmc", type=int, default=10000)
    parser.add_argument("--naxes", type=int, default=1000)
    parser.add_argument("--lmax", type=int, default=15)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("D_n GROUP SPECIFICITY TEST")
    print("=" * 70)

    # Load SMICA
    print(f"\n  Loading SMICA: {args.smica}")
    smica_map = hp.read_map(args.smica, field=0)
    nside_orig = hp.get_nside(smica_map)
    print(f"    NSIDE={nside_orig}")

    # Downgrade to NSIDE=256
    target_nside = 256 if args.lmax <= 150 else nside_orig
    if nside_orig != target_nside:
        print(f"    Downgrading to NSIDE={target_nside}")
        smica_map = hp.ud_grade(smica_map, target_nside)

    lmax = args.lmax
    print(f"  Extracting alm (lmax={lmax})...")
    alm_smica = hp.map2alm(smica_map, lmax=lmax)
    cl_smica = hp.alm2cl(alm_smica, lmax=lmax)

    # Load other maps for rank-1 test
    all_maps_alm = [alm_smica]
    map_names = ["SMICA"]
    for name, path in [("NILC", args.nilc), ("Commander", args.commander),
                       ("SEVEM", args.sevem)]:
        if path and Path(path).exists():
            print(f"  Loading {name}: {path}")
            m = hp.read_map(path, field=0)
            if hp.get_nside(m) != target_nside:
                m = hp.ud_grade(m, target_nside)
            a = hp.map2alm(m, lmax=lmax)
            all_maps_alm.append(a)
            map_names.append(name)
    print(f"  Maps loaded: {map_names}")

    # Verification: dn_fractions(n=3) matches d3_fractions_original
    print(f"\n  Verification: dn_fractions(n=3) vs original d3_fractions...")
    alm_d3 = rotate_to_axis(alm_smica, lmax, D3_ANTI_L, D3_ANTI_B)
    max_diff = 0.0
    for l in range(2, lmax + 1):
        fA1_orig, fA2_orig, fE_orig = d3_fractions_original(alm_d3, lmax, l)
        fracs_new = dn_fractions(alm_d3, lmax, l, 3)
        diff = abs(fA2_orig - fracs_new["A2"])
        max_diff = max(max_diff, diff)
    print(f"    Max |f_A2 difference|: {max_diff:.2e} {'PASS' if max_diff < 1e-10 else 'FAIL'}")

    groups = [3, 4, 5, 6]
    all_results = {}

    # ===================================================================
    # TEST A: ALL GROUPS AT D3 AXIS
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("TEST A: ALL GROUPS AT D3 AXIS (50.3, -64.9)")
    print(f"{'='*70}")

    for n in groups:
        result = run_dn_analysis(
            alm_smica, lmax, n,
            pole_l=D3_ANTI_L, pole_b=D3_ANTI_B,
            n_mc=args.nmc, n_axes=args.naxes,
            cl_fid=cl_smica, nside=target_nside,
            label=f"D{n}_at_D3_axis")

        key = f"d{n}_at_d3_axis"
        all_results[key] = result
        out_path = outdir / f"{key}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    # ===================================================================
    # TEST B: AXIS SEARCH FOR EACH GROUP
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("TEST B: AXIS SEARCH FOR EACH GROUP")
    print(f"{'='*70}")

    optimal_axes = {}
    for n in groups:
        search_result, opt_pole_l, opt_pole_b = axis_search(
            alm_smica, lmax, n, search_nside=16)

        key = f"d{n}_axis_search"
        all_results[key] = search_result
        out_path = outdir / f"{key}.json"
        with open(out_path, "w") as f:
            json.dump(search_result, f, indent=2)
        print(f"  Saved: {out_path}")
        optimal_axes[n] = (opt_pole_l, opt_pole_b)

    # ===================================================================
    # TEST B continued: FULL ANALYSIS AT OWN AXIS
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("TEST B: FULL ANALYSIS AT EACH GROUP'S OWN AXIS")
    print(f"{'='*70}")

    for n in groups:
        opt_pole_l, opt_pole_b = optimal_axes[n]
        result = run_dn_analysis(
            alm_smica, lmax, n,
            pole_l=opt_pole_l, pole_b=opt_pole_b,
            n_mc=args.nmc, n_axes=args.naxes,
            cl_fid=cl_smica, nside=target_nside,
            label=f"D{n}_at_own_axis")

        key = f"d{n}_at_d{n}_axis"
        all_results[key] = result
        out_path = outdir / f"{key}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    # ===================================================================
    # TEST C: RANK-1 FACTORIZATION
    # ===================================================================
    if len(all_maps_alm) >= 3:
        print(f"\n\n{'='*70}")
        print("TEST C: RANK-1 FACTORIZATION (4 Planck maps)")
        print(f"{'='*70}")

        rank1_results = {}
        for n in groups:
            # At D3 axis
            r1_d3 = rank1_test(all_maps_alm, lmax, n, D3_ANTI_L, D3_ANTI_B)
            # At own axis
            opt_pole_l, opt_pole_b = optimal_axes[n]
            r1_own = rank1_test(all_maps_alm, lmax, n, opt_pole_l, opt_pole_b)

            rank1_results[f"D{n}"] = {
                "at_d3_axis": r1_d3,
                "at_own_axis": r1_own,
            }
            print(f"  D{n}: rank1_frac at D3 axis = {r1_d3['rank1_frac']:.4f}, "
                  f"sigma ratio = {r1_d3['sigma_ratio']:.2f}")
            print(f"        rank1_frac at own axis = {r1_own['rank1_frac']:.4f}, "
                  f"sigma ratio = {r1_own['sigma_ratio']:.2f}")

        all_results["rank1_tests"] = rank1_results

    # ===================================================================
    # GRAMMAR COMPARISON
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("GRAMMAR SCORECARD")
    print(f"{'='*70}")

    grammar = {"comparison_type": "grammar_scorecard", "groups_tested": []}
    grammar["per_group"] = {}

    for n in groups:
        gname = f"D{n}"
        grammar["groups_tested"].append(gname)

        # Use own-axis result for grammar
        own_key = f"d{n}_at_d{n}_axis"
        d3_key = f"d{n}_at_d3_axis"

        r_own = all_results.get(own_key, {})
        r_d3 = all_results.get(d3_key, {})

        # Axis search
        search_key = f"d{n}_axis_search"
        search = all_results.get(search_key, {})

        # Grammar elements
        fp_own = r_own.get("A2_fisher_pte_low_l", 1.0)
        fp_d3 = r_d3.get("A2_fisher_pte_low_l", 1.0)
        fa_own = r_own.get("A2_fisher_axis_scramble_pte", 1.0)
        fa_d3 = r_d3.get("A2_fisher_axis_scramble_pte", 1.0)

        pg = r_own.get("parity_gate", {})
        pg_present = pg.get("odd_l_sum_delta_A2", 0) > pg.get("even_l_sum_delta_A2", 0)
        pg_ratio = pg.get("parity_asymmetry_ratio", 0)

        r_ea = r_own.get("inter_irrep_r_A2_E", 0)

        r1 = all_results.get("rank1_tests", {}).get(gname, {}).get("at_own_axis", {})
        r1_frac = r1.get("rank1_frac", 0)
        r1_ratio = r1.get("sigma_ratio", 0)

        # Xi persistent drift
        xi_list = r_own.get("xi_cumulative", [])
        xi_positive_at_15 = xi_list[-1]["xi"] > 0 if xi_list else False

        # Count grammar passes
        passes = 0
        if fp_own < 0.05:
            passes += 1
        if pg_present and pg_ratio > 0.5:
            passes += 1
        if r1_frac > 0.9:
            passes += 1
        if r_ea < -0.5:
            passes += 1
        if xi_positive_at_15:
            passes += 1

        grammar["per_group"][gname] = {
            "axis_own": search.get("optimal_axis", {}),
            "sep_from_d3": search.get("separation_from_d3_deg", 0),
            "A2_fisher_pte_at_d3": float(fp_d3),
            "A2_fisher_pte_at_own": float(fp_own),
            "axis_scramble_pte_at_d3": float(fa_d3),
            "axis_scramble_pte_at_own": float(fa_own),
            "parity_gate_present": bool(pg_present),
            "parity_ratio": float(pg_ratio),
            "rank1_fraction": float(r1_frac),
            "rank1_sigma_ratio": float(r1_ratio),
            "E_to_A2_funnel_r": float(r_ea),
            "xi_positive_drift": bool(xi_positive_at_15),
            "grammar_elements_passed": f"{passes}/5",
        }

        print(f"\n  {gname} (axis: {search.get('optimal_axis', {})}, "
              f"sep from D3: {search.get('separation_from_d3_deg', 0):.1f} deg):")
        print(f"    Fisher PTE (own axis): {fp_own:.6f}  {'PASS' if fp_own < 0.05 else 'FAIL'}")
        print(f"    Fisher PTE (D3 axis):  {fp_d3:.6f}")
        print(f"    Axis scramble (own):   {fa_own:.6f}")
        print(f"    Axis scramble (D3):    {fa_d3:.6f}")
        print(f"    Parity gate: {'YES' if pg_present else 'NO'} "
              f"(ratio={pg_ratio:+.4f})  "
              f"{'PASS' if pg_present and pg_ratio > 0.5 else 'FAIL'}")
        print(f"    Rank-1: {r1_frac:.4f} (sigma ratio={r1_ratio:.2f})  "
              f"{'PASS' if r1_frac > 0.9 else 'FAIL'}")
        print(f"    E->A2 funnel r: {r_ea:.4f}  "
              f"{'PASS' if r_ea < -0.5 else 'FAIL'}")
        print(f"    Xi drift: {'positive' if xi_positive_at_15 else 'negative'}  "
              f"{'PASS' if xi_positive_at_15 else 'FAIL'}")
        print(f"    GRAMMAR: {passes}/5")

    # Determine verdict
    d3_passes = int(grammar["per_group"]["D3"]["grammar_elements_passed"][0])
    others_max = max(int(grammar["per_group"][f"D{n}"]["grammar_elements_passed"][0])
                     for n in [4, 5, 6])

    if d3_passes >= 4 and others_max <= 2:
        verdict = "D3 IS uniquely selected by the full grammar"
    elif d3_passes >= 4 and others_max == 3:
        verdict = "D3 is PARTIALLY special — strongest grammar, but others show fragments"
    elif d3_passes <= others_max:
        verdict = "D3 is NOT uniquely special — other groups show comparable grammar"
    else:
        verdict = "D3 has the best grammar but specificity is moderate"

    grammar["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    grammar_path = outdir / "grammar_comparison.json"
    with open(grammar_path, "w") as f:
        json.dump(grammar, f, indent=2)
    print(f"  Saved: {grammar_path}")

    # ===================================================================
    # SUMMARY TABLE
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Group':<8s} {'Axis sep':>8s} {'Fisher D3':>10s} {'Fisher own':>10s} "
          f"{'Axis scr':>10s} {'Parity':>8s} {'R1 frac':>8s} {'r(A2,E)':>8s} {'Score':>6s}")
    print(f"  {'-'*80}")

    for n in groups:
        g = grammar["per_group"][f"D{n}"]
        print(f"  D{n:<7d} {g['sep_from_d3']:7.1f}° "
              f"{g['A2_fisher_pte_at_d3']:10.6f} {g['A2_fisher_pte_at_own']:10.6f} "
              f"{g['axis_scramble_pte_at_own']:10.6f} "
              f"{g['parity_ratio']:+7.3f} {g['rank1_fraction']:8.4f} "
              f"{g['E_to_A2_funnel_r']:+7.4f} {g['grammar_elements_passed']:>6s}")


if __name__ == "__main__":
    main()
