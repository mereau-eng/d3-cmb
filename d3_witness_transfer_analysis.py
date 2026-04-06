#!/usr/bin/env python3
"""
D3 Witness-Sector Transfer Analysis (Batch 3).

Tests whether the observed parity × D3-residue structure factorizes into
a binary gate (odd/even) ⊗ D3 residue routing (l mod 3), as predicted
by the witness-sector / Mandala framework.

Tests implemented:
  1. Conditional transfer theorem: 2×3 table, parity splits per residue class
  2. Irrep-parity transfer matrix: 3-vector (dA1, dA2, dE) per parity branch
  3. First-allowed-mode gate: capacity-normalized onset at l=3
  4. 2×3 factorization (SVD): rank-1 test of M(L)
  5. Generation-shadow: 3-level hierarchy of significant multipoles
  6. Raw-power control: odd/even in C_l vs f_A2 vs delta_A2
  7. Cross-map eigenvector stability: SVD vectors across 4 maps

Usage:
  python d3_witness_transfer_analysis.py \
    --results extended_results \
    --fits map1.fits [map2.fits ...] \
    --lmax 150
"""

import numpy as np
import healpy as hp
import json, os, argparse, glob, time


# =====================================================================
# DATA LOADING
# =====================================================================

def load_batch1_results(results_dir):
    """Load all per-map extended result JSONs from Batch 1."""
    maps = {}
    for name in ['smica', 'nilc', 'sevem', 'commander']:
        pattern = os.path.join(results_dir, f'd3_extended_{name}_lmax*.json')
        matches = glob.glob(pattern)
        if matches:
            with open(matches[0]) as f:
                maps[name.upper()] = json.load(f)
    return maps


def expected_fractions(l):
    """Expected irrep fractions under isotropy for multipole l."""
    n_a2_pairs = l // 3  # floor(l/3): number of m>0 with m%3==0
    dof_A1 = 1 + n_a2_pairs   # m=0 (1 real DOF) + Re of m%3==0 pairs
    dof_A2 = n_a2_pairs        # Im of m%3==0 pairs
    dof_E = 2 * (l - n_a2_pairs)  # all m%3!=0 pairs, 2 DOF each
    total = 2 * l + 1
    return dof_A1 / total, dof_A2 / total, dof_E / total


def build_per_l_table(data, lmax=150):
    """Extract per-l quantities including all three delta_irrep values."""
    per_l = data['per_l']
    table = []
    for l in range(2, lmax + 1):
        l_str = str(l)
        if l_str not in per_l:
            continue
        d = per_l[l_str]
        ef_A1, ef_A2, ef_E = expected_fractions(l)
        delta_A1 = d['f_A1'] - ef_A1
        delta_A2 = d['f_A2'] - ef_A2
        delta_E = d['f_E'] - ef_E
        table.append({
            'l': l,
            'parity': 'odd' if l % 2 == 1 else 'even',
            'r': l % 3,
            'f_A1': d['f_A1'], 'f_A2': d['f_A2'], 'f_E': d['f_E'],
            'ef_A1': ef_A1, 'ef_A2': ef_A2, 'ef_E': ef_E,
            'delta_A1': delta_A1, 'delta_A2': delta_A2, 'delta_E': delta_E,
            'pte_A2': d['pte_A2'], 'z_A2': d['z_A2'],
            'd_A2': l // 3,  # dimension of A2 irrep
        })
    return table


# =====================================================================
# TEST 1: CONDITIONAL TRANSFER THEOREM
# =====================================================================

def test1_conditional_transfer(table, label):
    """Build 2×3 transfer table and compute conditional differences."""
    print(f"\n{'='*72}")
    print(f"TEST 1: CONDITIONAL TRANSFER THEOREM — {label}")
    print(f"{'='*72}")

    # Build cumulative T_{epsilon, r}(L) for L = lmax
    T = {}
    for eps in ['odd', 'even']:
        for r in [0, 1, 2]:
            T[(eps, r)] = sum(d['delta_A2'] for d in table
                              if d['parity'] == eps and d['r'] == r)

    print(f"\n  Transfer table T_{{parity, r}} (cumulative, L={table[-1]['l']}):")
    print(f"  {'':>8} {'r=0':>12} {'r=1':>12} {'r=2':>12} {'Row sum':>12}")
    print(f"  {'-'*52}")
    for eps in ['odd', 'even']:
        row = [T[(eps, r)] for r in [0, 1, 2]]
        print(f"  {eps:>8} {row[0]:>+12.5f} {row[1]:>+12.5f} {row[2]:>+12.5f} {sum(row):>+12.5f}")
    print(f"  {'Col sum':>8}", end="")
    for r in [0, 1, 2]:
        cs = T[('odd', r)] + T[('even', r)]
        print(f" {cs:>+12.5f}", end="")
    print()

    # Conditional differences
    print(f"\n  Parity difference per residue class:")
    print(f"  Delta^(r)_parity = T_odd,r - T_even,r")
    for r in [0, 1, 2]:
        diff = T[('odd', r)] - T[('even', r)]
        print(f"    r={r}: {diff:>+.5f}")

    print(f"\n  Residue preference per parity branch:")
    print(f"  Delta^(eps)_residue = T_eps,0 - T_eps,2")
    for eps in ['odd', 'even']:
        diff = T[(eps, 0)] - T[(eps, 2)]
        print(f"    {eps}: {diff:>+.5f}")

    # Return the table for later use
    return T


def test1_cumulative_evolution(table, label):
    """Show how the 2×3 table evolves with L."""
    print(f"\n  Cumulative T_{{odd,0}}(L) and T_{{even,0}}(L) evolution:")
    print(f"  {'L':>6} {'T_odd,0':>10} {'T_even,0':>10} {'T_odd,1':>10} {'T_even,1':>10} {'T_odd,2':>10} {'T_even,2':>10}")
    print(f"  {'-'*66}")

    cum = {(eps, r): 0.0 for eps in ['odd', 'even'] for r in [0, 1, 2]}
    checkpoints = [10, 15, 20, 30, 50, 75, 100, 125, 150]
    for d in table:
        cum[(d['parity'], d['r'])] += d['delta_A2']
        if d['l'] in checkpoints:
            print(f"  {d['l']:>6}", end="")
            for eps in ['odd', 'even']:
                for r in [0, 1, 2]:
                    print(f" {cum[(eps, r)]:>+10.4f}", end="")
            print()


# =====================================================================
# TEST 2: IRREP-PARITY TRANSFER MATRIX
# =====================================================================

def test2_irrep_parity(table, label):
    """Track 3-vector (dA1, dA2, dE) per parity branch."""
    print(f"\n{'='*72}")
    print(f"TEST 2: IRREP-PARITY TRANSFER MATRIX — {label}")
    print(f"{'='*72}")

    # Cumulative vectors
    Delta_odd = np.zeros(3)
    Delta_even = np.zeros(3)

    # Track direction stability of G(L) = Delta_odd - Delta_even
    G_history = []
    L_history = []

    for d in table:
        vec = np.array([d['delta_A1'], d['delta_A2'], d['delta_E']])
        if d['parity'] == 'odd':
            Delta_odd += vec
        else:
            Delta_even += vec

        if d['l'] >= 10:  # start tracking once enough data
            G = Delta_odd - Delta_even
            G_norm = np.linalg.norm(G)
            if G_norm > 1e-10:
                G_hat = G / G_norm
                G_history.append(G_hat.copy())
                L_history.append(d['l'])

    G_final = Delta_odd - Delta_even
    G_norm_final = np.linalg.norm(G_final)
    G_hat_final = G_final / G_norm_final if G_norm_final > 0 else G_final

    print(f"\n  Cumulative irrep vectors at L={table[-1]['l']}:")
    print(f"  {'':>10} {'dA1':>10} {'dA2':>10} {'dE':>10} {'|vec|':>10}")
    print(f"  {'-'*46}")
    for name, vec in [('odd', Delta_odd), ('even', Delta_even), ('G=diff', G_final)]:
        print(f"  {name:>10} {vec[0]:>+10.5f} {vec[1]:>+10.5f} {vec[2]:>+10.5f} {np.linalg.norm(vec):>10.5f}")

    print(f"\n  G direction (unit): ({G_hat_final[0]:+.4f}, {G_hat_final[1]:+.4f}, {G_hat_final[2]:+.4f})")

    # Conservation check
    print(f"\n  Conservation check (dA1+dA2+dE per row):")
    print(f"    odd:  {np.sum(Delta_odd):+.2e}")
    print(f"    even: {np.sum(Delta_even):+.2e}")
    print(f"    G:    {np.sum(G_final):+.2e}")

    # Direction stability: cosine similarity between G(L) and G(L_max)
    print(f"\n  Direction stability of G(L) (cosine with G_final):")
    print(f"  {'L':>6} {'cos(G(L), G_final)':>20} {'|G(L)|':>10}")
    print(f"  {'-'*40}")
    checkpoints = [10, 15, 20, 30, 50, 75, 100, 125, 150]
    for i, (L, G_hat) in enumerate(zip(L_history, G_history)):
        if L in checkpoints:
            cos_sim = np.dot(G_hat, G_hat_final)
            # Reconstruct magnitude
            D_o = np.zeros(3)
            D_e = np.zeros(3)
            for d in table:
                if d['l'] > L:
                    break
                vec = np.array([d['delta_A1'], d['delta_A2'], d['delta_E']])
                if d['parity'] == 'odd':
                    D_o += vec
                else:
                    D_e += vec
            G_mag = np.linalg.norm(D_o - D_e)
            print(f"  {L:>6} {cos_sim:>+20.4f} {G_mag:>10.5f}")

    return G_hat_final


# =====================================================================
# TEST 3: FIRST-ALLOWED-MODE GATE TEST
# =====================================================================

def test3_first_allowed_mode(table, label):
    """Capacity-normalized onset at l=3 for A2."""
    print(f"\n{'='*72}")
    print(f"TEST 3: FIRST-ALLOWED-MODE GATE TEST — {label}")
    print(f"{'='*72}")

    print(f"\n  A2 dimension and capacity fraction by l:")
    print(f"  {'l':>4} {'d_A2':>6} {'d_A2/(2l+1)':>14} {'delta_A2':>10} {'cap-norm dA2':>14} {'z_A2':>8} {'parity':>8}")
    print(f"  {'-'*68}")

    cap_norm_values = []
    for d in table:
        if d['l'] > 20:
            break
        l = d['l']
        d_A2 = d['d_A2']
        cap_frac = d_A2 / (2 * l + 1) if d_A2 > 0 else 0
        cap_norm = d['delta_A2'] / cap_frac if cap_frac > 0 else float('nan')
        cap_norm_values.append({'l': l, 'cap_norm': cap_norm, 'parity': d['parity']})
        print(f"  {l:>4} {d_A2:>6} {cap_frac:>14.4f} {d['delta_A2']:>+10.5f} "
              f"{cap_norm:>+14.4f} {d['z_A2']:>+8.2f} {d['parity']:>8}")

    # Step-function test: ratio of l=3 cap-norm to mean of l=4..15
    l3 = next(d for d in cap_norm_values if d['l'] == 3)
    others = [d['cap_norm'] for d in cap_norm_values
              if d['l'] >= 4 and d['l'] <= 15 and not np.isnan(d['cap_norm'])]

    if others:
        mean_others = np.mean(others)
        std_others = np.std(others)
        step_ratio = l3['cap_norm'] / mean_others if mean_others != 0 else float('inf')
        step_z = (l3['cap_norm'] - mean_others) / std_others if std_others > 0 else 0

        print(f"\n  Step-function analysis:")
        print(f"    l=3 cap-normalized excess: {l3['cap_norm']:+.4f}")
        print(f"    Mean l=4..15 cap-normalized: {mean_others:+.4f}")
        print(f"    Std l=4..15: {std_others:.4f}")
        print(f"    Step ratio (l=3 / mean): {step_ratio:.2f}")
        print(f"    Step z-score: {step_z:+.2f}")

    # Capacity-normalized by parity
    print(f"\n  Capacity-normalized excess by parity (l=3..20, d_A2 > 0):")
    odd_cn = [d['cap_norm'] for d in cap_norm_values
              if d['parity'] == 'odd' and not np.isnan(d['cap_norm'])]
    even_cn = [d['cap_norm'] for d in cap_norm_values
               if d['parity'] == 'even' and not np.isnan(d['cap_norm'])]
    if odd_cn:
        print(f"    Odd mean:  {np.mean(odd_cn):+.4f}")
    if even_cn:
        print(f"    Even mean: {np.mean(even_cn):+.4f}")


# =====================================================================
# TEST 4: 2×3 FACTORIZATION (SVD) — THE KEY TEST
# =====================================================================

def test4_svd_factorization(table, label):
    """SVD of the 2×3 transfer matrix M(L)."""
    print(f"\n{'='*72}")
    print(f"TEST 4: 2×3 FACTORIZATION (SVD) — {label}")
    print(f"{'='*72}")

    # Build cumulative M(L) and do SVD at multiple L values
    M_cum = np.zeros((2, 3))  # rows: odd=0, even=1; cols: r=0,1,2
    parity_idx = {'odd': 0, 'even': 1}

    results_by_L = []
    checkpoints = list(range(10, 151, 5))

    for d in table:
        M_cum[parity_idx[d['parity']], d['r']] += d['delta_A2']

        if d['l'] in checkpoints:
            M = M_cum.copy()
            U, s, Vt = np.linalg.svd(M, full_matrices=True)
            total_var = np.sum(s**2)
            rank1_frac = s[0]**2 / total_var if total_var > 0 else 0

            results_by_L.append({
                'L': d['l'],
                'M': M.copy(),
                's': s.copy(),
                'u': U[:, 0].copy(),
                'v': Vt[0, :].copy(),
                'rank1_frac': rank1_frac,
            })

    # Print evolution
    print(f"\n  Rank-1 fraction (sigma1^2 / sum(sigma^2)) vs L:")
    print(f"  {'L':>6} {'sigma1':>10} {'sigma2':>10} {'rank1%':>10} {'u (odd,even)':>20} {'v (r=0,r=1,r=2)':>28}")
    print(f"  {'-'*88}")

    for r in results_by_L:
        if r['L'] % 10 == 0 or r['L'] in [15, 25]:
            u_str = f"({r['u'][0]:+.3f}, {r['u'][1]:+.3f})"
            v_str = f"({r['v'][0]:+.3f}, {r['v'][1]:+.3f}, {r['v'][2]:+.3f})"
            print(f"  {r['L']:>6} {r['s'][0]:>10.5f} {r['s'][1]:>10.5f} "
                  f"{r['rank1_frac']*100:>9.1f}% {u_str:>20} {v_str:>28}")

    # Final matrix and SVD
    M_final = M_cum.copy()
    U_f, s_f, Vt_f = np.linalg.svd(M_final, full_matrices=True)
    rank1_final = s_f[0]**2 / np.sum(s_f**2) if np.sum(s_f**2) > 0 else 0

    print(f"\n  Final M(L={table[-1]['l']}):")
    print(f"         r=0        r=1        r=2")
    print(f"  odd   {M_final[0,0]:>+10.5f} {M_final[0,1]:>+10.5f} {M_final[0,2]:>+10.5f}")
    print(f"  even  {M_final[1,0]:>+10.5f} {M_final[1,1]:>+10.5f} {M_final[1,2]:>+10.5f}")

    print(f"\n  SVD:")
    print(f"    sigma1 = {s_f[0]:.5f}")
    print(f"    sigma2 = {s_f[1]:.5f}")
    print(f"    Rank-1 fraction: {rank1_final*100:.1f}%")
    print(f"    sigma1/sigma2 ratio: {s_f[0]/s_f[1]:.1f}" if s_f[1] > 1e-10 else "    sigma2 ≈ 0 (exact rank 1)")

    u1 = U_f[:, 0]
    v1 = Vt_f[0, :]
    print(f"\n    Dominant left vector u (binary gate):")
    print(f"      u = ({u1[0]:+.4f}, {u1[1]:+.4f})")
    print(f"      Interpretation: odd={u1[0]:+.4f}, even={u1[1]:+.4f}")
    expected_u = np.array([-1/np.sqrt(2), 1/np.sqrt(2)])
    # Check if u is close to (±1, ∓1)/sqrt(2) — i.e., odd positive / even negative
    # Need to handle sign ambiguity of SVD
    if u1[0] > 0:  # odd positive
        print(f"      Sign: odd POSITIVE, even NEGATIVE ✓")
    else:
        print(f"      Sign: odd NEGATIVE, even POSITIVE (flipped)")
        u1 = -u1
        v1 = -v1

    print(f"\n    Dominant right vector v (D3 residue routing):")
    print(f"      v = ({v1[0]:+.4f}, {v1[1]:+.4f}, {v1[2]:+.4f})")
    print(f"      Interpretation: r=0 ({v1[0]:+.4f}), r=1 ({v1[1]:+.4f}), r=2 ({v1[2]:+.4f})")

    # Rank-1 reconstruction error
    M_rank1 = s_f[0] * np.outer(U_f[:, 0], Vt_f[0, :])
    resid = M_final - M_rank1
    rel_error = np.linalg.norm(resid) / np.linalg.norm(M_final) if np.linalg.norm(M_final) > 0 else 0
    print(f"\n    Rank-1 reconstruction error: {rel_error*100:.1f}% (Frobenius norm)")

    return {
        'M': M_final,
        'u': u1,  # sign-normalized: odd positive
        'v': v1,
        's': s_f,
        'rank1_frac': rank1_final,
        'results_by_L': results_by_L,
    }


# =====================================================================
# TEST 5: GENERATION-SHADOW TEST
# =====================================================================

def test5_generation_shadow(table, label):
    """Classify significant multipoles into hierarchy levels."""
    print(f"\n{'='*72}")
    print(f"TEST 5: GENERATION-SHADOW TEST — {label}")
    print(f"{'='*72}")

    # Sort by significance
    sig_multi = sorted(
        [d for d in table if d['pte_A2'] < 0.10],
        key=lambda d: d['pte_A2']
    )

    print(f"\n  Significant multipoles (PTE < 0.10), ranked:")
    print(f"  {'l':>4} {'parity':>8} {'r=l%3':>6} {'delta_A2':>10} {'PTE':>10} {'z':>8} {'tier':>8}")
    print(f"  {'-'*58}")

    tiers = {'primary': [], 'secondary': [], 'echo': []}

    for d in sig_multi:
        if d['pte_A2'] < 0.005:
            tier = 'primary'
        elif d['pte_A2'] < 0.05:
            tier = 'secondary'
        else:
            tier = 'echo'
        tiers[tier].append(d)
        print(f"  {d['l']:>4} {d['parity']:>8} {d['r']:>6} {d['delta_A2']:>+10.5f} "
              f"{d['pte_A2']:>10.4f} {d['z_A2']:>+8.2f} {tier:>8}")

    # Summarize tiers
    print(f"\n  Tier summary:")
    for tier_name in ['primary', 'secondary', 'echo']:
        members = tiers[tier_name]
        if members:
            ls = [d['l'] for d in members]
            parities = [d['parity'] for d in members]
            residues = [d['r'] for d in members]
            print(f"    {tier_name:>10}: l = {ls}")
            print(f"               parities: {parities}")
            print(f"               residues: {residues}")

    # Check D3 stratum mapping
    print(f"\n  D3 stratum mapping check:")
    print(f"    Primary (core/3rd-gen):   l mod 3 = {[d['r'] for d in tiers['primary']]}")
    print(f"    Secondary (edge/2nd-gen): l mod 3 = {[d['r'] for d in tiers['secondary']]}")
    print(f"    Echo (vertex/1st-gen):    l mod 3 = {[d['r'] for d in tiers['echo']]}")

    # Parity distribution within tiers
    print(f"\n  Parity distribution within tiers:")
    for tier_name in ['primary', 'secondary', 'echo']:
        members = tiers[tier_name]
        n_odd = sum(1 for d in members if d['parity'] == 'odd')
        n_even = sum(1 for d in members if d['parity'] == 'even')
        total = len(members)
        if total > 0:
            print(f"    {tier_name:>10}: {n_odd} odd / {n_even} even (out of {total})")

    return tiers


# =====================================================================
# TEST 6: ODD/EVEN RAW-POWER CONTROL
# =====================================================================

def test6_raw_power_control(table, fits_paths, label, lmax=150):
    """Compare odd/even statistics for C_l, P_A2, f_A2, delta_A2."""
    print(f"\n{'='*72}")
    print(f"TEST 6: ODD/EVEN RAW-POWER CONTROL — {label}")
    print(f"{'='*72}")

    # Find the right FITS file for this map label
    fpath = None
    for fp in fits_paths:
        if label.lower() in os.path.basename(fp).lower():
            fpath = fp
            break
    if fpath is None:
        fpath = fits_paths[0] if fits_paths else None

    if fpath is None or not os.path.exists(fpath):
        print(f"  WARNING: No FITS file found for {label}, skipping C_l computation")
        return None

    # Load map and compute C_l
    sky_map = hp.read_map(fpath, field=0)
    if hp.get_nside(sky_map) > 256 and lmax <= 384:
        sky_map = hp.ud_grade(sky_map, 256)
    cl = hp.anafast(sky_map, lmax=lmax)

    # Build comparison table
    odd_stats = {'C_l': [], 'P_A2': [], 'f_A2': [], 'delta_A2': []}
    even_stats = {'C_l': [], 'P_A2': [], 'f_A2': [], 'delta_A2': []}

    for d in table:
        l = d['l']
        if l < 2 or l > lmax:
            continue
        C_l = cl[l]
        P_A2 = d['f_A2'] * C_l

        target = odd_stats if d['parity'] == 'odd' else even_stats
        target['C_l'].append(C_l)
        target['P_A2'].append(P_A2)
        target['f_A2'].append(d['f_A2'])
        target['delta_A2'].append(d['delta_A2'])

    # Compute means and test
    print(f"\n  Odd/even mean comparison:")
    print(f"  {'Quantity':>14} {'Mean(odd)':>14} {'Mean(even)':>14} {'Diff':>14} {'Ratio':>10}")
    print(f"  {'-'*68}")

    results = {}
    for key in ['C_l', 'P_A2', 'f_A2', 'delta_A2']:
        m_odd = np.mean(odd_stats[key])
        m_even = np.mean(even_stats[key])
        diff = m_odd - m_even
        ratio = m_odd / m_even if m_even != 0 else float('inf')
        results[key] = {'mean_odd': m_odd, 'mean_even': m_even, 'diff': diff}
        print(f"  {key:>14} {m_odd:>14.6f} {m_even:>14.6f} {diff:>+14.6f} {ratio:>10.3f}")

    # Permutation test for each quantity
    rng = np.random.default_rng(42)
    n_perm = 50000
    print(f"\n  Permutation PTE (N={n_perm}, two-sided |diff|):")
    for key in ['C_l', 'P_A2', 'f_A2', 'delta_A2']:
        all_vals = np.array(odd_stats[key] + even_stats[key])
        n_odd = len(odd_stats[key])
        obs_diff = abs(np.mean(odd_stats[key]) - np.mean(even_stats[key]))
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(all_vals)
            perm_diff = abs(np.mean(perm[:n_odd]) - np.mean(perm[n_odd:]))
            if perm_diff >= obs_diff:
                count += 1
        pte = count / n_perm
        sig = " *" if pte < 0.05 else (" **" if pte < 0.01 else "")
        print(f"    {key:>14}: PTE = {pte:.4f}{sig}")

    print(f"\n  Interpretation: if delta_A2 is most significant and C_l is not,")
    print(f"  the binary gate acts on ROUTING, not total power.")

    return results


# =====================================================================
# TEST 7: CROSS-MAP EIGENVECTOR STABILITY
# =====================================================================

def test7_cross_map_stability(svd_results):
    """Compare SVD vectors across maps."""
    print(f"\n{'='*72}")
    print("TEST 7: CROSS-MAP EIGENVECTOR STABILITY")
    print(f"{'='*72}")

    labels = list(svd_results.keys())
    n = len(labels)

    # Compare u vectors (binary gate)
    print(f"\n  Left singular vector u (binary gate) — pairwise cosine similarity:")
    print(f"  {'':>12}", end="")
    for lab in labels:
        print(f" {lab:>12}", end="")
    print()
    for i, l1 in enumerate(labels):
        print(f"  {l1:>12}", end="")
        for j, l2 in enumerate(labels):
            cos = np.dot(svd_results[l1]['u'], svd_results[l2]['u'])
            print(f" {cos:>+12.4f}", end="")
        print()

    # Compare v vectors (D3 residue routing)
    print(f"\n  Right singular vector v (D3 routing) — pairwise cosine similarity:")
    print(f"  {'':>12}", end="")
    for lab in labels:
        print(f" {lab:>12}", end="")
    print()
    for i, l1 in enumerate(labels):
        print(f"  {l1:>12}", end="")
        for j, l2 in enumerate(labels):
            cos = np.dot(svd_results[l1]['v'], svd_results[l2]['v'])
            print(f" {cos:>+12.4f}", end="")
        print()

    # Print all u and v vectors for comparison
    print(f"\n  All u vectors (sign-normalized: odd positive):")
    for lab in labels:
        u = svd_results[lab]['u']
        print(f"    {lab:>12}: u = ({u[0]:+.4f}, {u[1]:+.4f})")

    print(f"\n  All v vectors:")
    for lab in labels:
        v = svd_results[lab]['v']
        print(f"    {lab:>12}: v = ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")

    # Rank-1 fractions across maps
    print(f"\n  Rank-1 fraction across maps:")
    for lab in labels:
        r1 = svd_results[lab]['rank1_frac']
        print(f"    {lab:>12}: {r1*100:.1f}%")

    # Mean u and v across maps
    u_all = np.array([svd_results[lab]['u'] for lab in labels])
    v_all = np.array([svd_results[lab]['v'] for lab in labels])
    u_mean = np.mean(u_all, axis=0)
    v_mean = np.mean(v_all, axis=0)
    u_std = np.std(u_all, axis=0)
    v_std = np.std(v_all, axis=0)

    print(f"\n  Mean ± std across maps:")
    print(f"    u = ({u_mean[0]:+.4f}±{u_std[0]:.4f}, {u_mean[1]:+.4f}±{u_std[1]:.4f})")
    print(f"    v = ({v_mean[0]:+.4f}±{v_std[0]:.4f}, {v_mean[1]:+.4f}±{v_std[1]:.4f}, {v_mean[2]:+.4f}±{v_std[2]:.4f})")

    # Rank-1 stability: how does rank-1 fraction evolve across L for each map?
    print(f"\n  Rank-1 fraction evolution (% at selected L):")
    print(f"  {'L':>6}", end="")
    for lab in labels:
        print(f" {lab:>12}", end="")
    print()
    # Use the results_by_L from each map
    target_Ls = [20, 30, 50, 75, 100, 150]
    for target_L in target_Ls:
        print(f"  {target_L:>6}", end="")
        for lab in labels:
            found = None
            for r in svd_results[lab]['results_by_L']:
                if r['L'] == target_L:
                    found = r
                    break
            if found:
                print(f" {found['rank1_frac']*100:>11.1f}%", end="")
            else:
                print(f" {'—':>12}", end="")
        print()


# =====================================================================
# TEST 2b: IRREP G-VECTOR STABILITY ACROSS MAPS
# =====================================================================

def test2b_G_vector_stability(G_vectors):
    """Compare the redistribution eigenvector G across maps."""
    print(f"\n{'='*72}")
    print("TEST 2b: REDISTRIBUTION EIGENVECTOR G STABILITY ACROSS MAPS")
    print(f"{'='*72}")

    labels = list(G_vectors.keys())

    print(f"\n  G-hat vectors across maps:")
    for lab in labels:
        g = G_vectors[lab]
        print(f"    {lab:>12}: ({g[0]:+.4f}, {g[1]:+.4f}, {g[2]:+.4f})")

    print(f"\n  Pairwise cosine similarity:")
    print(f"  {'':>12}", end="")
    for lab in labels:
        print(f" {lab:>12}", end="")
    print()
    for l1 in labels:
        print(f"  {l1:>12}", end="")
        for l2 in labels:
            cos = np.dot(G_vectors[l1], G_vectors[l2])
            print(f" {cos:>+12.4f}", end="")
        print()

    g_all = np.array([G_vectors[lab] for lab in labels])
    g_mean = np.mean(g_all, axis=0)
    g_mean_hat = g_mean / np.linalg.norm(g_mean)
    g_std = np.std(g_all, axis=0)
    print(f"\n  Mean G-hat: ({g_mean_hat[0]:+.4f}, {g_mean_hat[1]:+.4f}, {g_mean_hat[2]:+.4f})")
    print(f"  Std:        ({g_std[0]:.4f}, {g_std[1]:.4f}, {g_std[2]:.4f})")


# =====================================================================
# MASTER SUMMARY
# =====================================================================

def print_master_summary(svd_results, G_vectors):
    """Print the master summary testing the core prediction."""
    print(f"\n{'='*72}")
    print("MASTER SUMMARY: 2×3 FACTORIZATION TEST")
    print(f"{'='*72}")

    labels = list(svd_results.keys())

    print(f"\n  Core prediction: M(L) ≈ sigma1 * u * v^T")
    print(f"    u ≈ (odd+, even-)  i.e. binary gate")
    print(f"    v ≈ (r=0 dominant, r=1 weak, r=2 draining)  i.e. D3 routing")
    print(f"    M approximately rank 1 over wide L range")

    print(f"\n  {'Map':>12} {'Rank-1%':>10} {'u_odd':>8} {'u_even':>8} {'v_r0':>8} {'v_r1':>8} {'v_r2':>8}")
    print(f"  {'-'*60}")
    for lab in labels:
        r = svd_results[lab]
        print(f"  {lab:>12} {r['rank1_frac']*100:>9.1f}% "
              f"{r['u'][0]:>+8.4f} {r['u'][1]:>+8.4f} "
              f"{r['v'][0]:>+8.4f} {r['v'][1]:>+8.4f} {r['v'][2]:>+8.4f}")

    # Assess success criteria
    all_rank1 = [svd_results[lab]['rank1_frac'] for lab in labels]
    mean_rank1 = np.mean(all_rank1)

    u_all = np.array([svd_results[lab]['u'] for lab in labels])
    v_all = np.array([svd_results[lab]['v'] for lab in labels])

    # u vectors: should all have u[0]>0 (odd positive) and u[1]<0 (even negative)
    u_correct = all(svd_results[lab]['u'][0] > 0 and svd_results[lab]['u'][1] < 0 for lab in labels)

    # v vectors: should have v[0] dominant
    v_r0_dominant = all(svd_results[lab]['v'][0] > abs(svd_results[lab]['v'][1]) and
                        svd_results[lab]['v'][0] > abs(svd_results[lab]['v'][2]) for lab in labels)

    # Cross-map stability of u
    u_min_cos = min(np.dot(svd_results[l1]['u'], svd_results[l2]['u'])
                    for i, l1 in enumerate(labels) for l2 in labels[i+1:])
    # Cross-map stability of v
    v_min_cos = min(np.dot(svd_results[l1]['v'], svd_results[l2]['v'])
                    for i, l1 in enumerate(labels) for l2 in labels[i+1:])

    print(f"\n  SUCCESS CRITERIA:")
    print(f"    1. M(L) approximately rank 1:    mean {mean_rank1*100:.1f}%  {'✓' if mean_rank1 > 0.80 else '✗'}")
    print(f"    2. u = (odd+, even-):            {u_correct}  {'✓' if u_correct else '✗'}")
    print(f"    3. v: r=0 dominant:              {v_r0_dominant}  {'✓' if v_r0_dominant else '✗'}")
    print(f"    4. Stable u across maps:         min cos = {u_min_cos:+.4f}  {'✓' if u_min_cos > 0.95 else '✗'}")
    print(f"    5. Stable v across maps:         min cos = {v_min_cos:+.4f}  {'✓' if v_min_cos > 0.90 else '✗'}")

    n_pass = sum([mean_rank1 > 0.80, u_correct, v_r0_dominant, u_min_cos > 0.95, v_min_cos > 0.90])
    print(f"\n  OVERALL: {n_pass}/5 criteria met")

    if n_pass == 5:
        print(f"\n  VERDICT: FULL FACTORIZATION CONFIRMED")
        print(f"  The 2×3 grammar factorizes as (binary gate) ⊗ (D3 residue routing)")
    elif n_pass >= 3:
        print(f"\n  VERDICT: PARTIAL FACTORIZATION")
        print(f"  Evidence for factorization structure, but not all criteria met")
    else:
        print(f"\n  VERDICT: FACTORIZATION NOT SUPPORTED")


# =====================================================================
# MAIN
# =====================================================================

def run_all(results_dir, fits_paths, lmax=150):
    t0 = time.time()

    # Load Batch 1 results
    maps_data = load_batch1_results(results_dir)
    if not maps_data:
        print("ERROR: No Batch 1 result files found")
        return

    print(f"{'#'*72}")
    print(f"# D3 WITNESS-SECTOR TRANSFER ANALYSIS (BATCH 3)")
    print(f"# Maps: {', '.join(maps_data.keys())}")
    print(f"# lmax: {lmax}")
    print(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*72}")

    svd_results = {}
    G_vectors = {}
    all_output = {}

    for label, data in maps_data.items():
        table = build_per_l_table(data, lmax)

        # Test 1
        T = test1_conditional_transfer(table, label)
        test1_cumulative_evolution(table, label)

        # Test 2
        G_hat = test2_irrep_parity(table, label)
        G_vectors[label] = G_hat

        # Test 3
        test3_first_allowed_mode(table, label)

        # Test 4 (THE KEY TEST)
        svd_result = test4_svd_factorization(table, label)
        svd_results[label] = svd_result

        # Test 5
        tiers = test5_generation_shadow(table, label)

        # Test 6
        test6_result = test6_raw_power_control(table, fits_paths, label, lmax)

        # Collect for JSON output
        all_output[label] = {
            'transfer_table': {f"{eps}_{r}": float(T[(eps, r)])
                               for eps in ['odd', 'even'] for r in [0, 1, 2]},
            'svd': {
                'rank1_frac': float(svd_result['rank1_frac']),
                'sigma1': float(svd_result['s'][0]),
                'sigma2': float(svd_result['s'][1]),
                'u': svd_result['u'].tolist(),
                'v': svd_result['v'].tolist(),
                'M': svd_result['M'].tolist(),
            },
            'G_vector': G_hat.tolist(),
            'tiers': {
                tier: [d['l'] for d in members]
                for tier, members in tiers.items()
            },
        }

    # Cross-map tests
    test7_cross_map_stability(svd_results)
    test2b_G_vector_stability(G_vectors)
    print_master_summary(svd_results, G_vectors)

    # Save results
    outfile = os.path.join(results_dir, 'witness_transfer_analysis.json')
    with open(outfile, 'w') as f:
        json.dump(all_output, f, indent=2)
    print(f"\n  Results saved to {outfile}")
    print(f"  Total time: {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='D3 Witness-Sector Transfer Analysis')
    parser.add_argument('--results', required=True, help='Directory with Batch 1 extended result JSONs')
    parser.add_argument('--fits', nargs='+', default=[], help='FITS map paths (for Test 6)')
    parser.add_argument('--lmax', type=int, default=150)
    args = parser.parse_args()
    run_all(args.results, args.fits, args.lmax)


if __name__ == '__main__':
    main()
