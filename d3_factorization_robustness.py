#!/usr/bin/env python3
"""
Robustness test for the 2×3 factorization.

The killer question: does M(L) ≈ sigma1 * u * v^T survive after
removing the dominant low-l multipoles (l=3, 7, 9)?

Tests multiple exclusion sets:
  1. Full (no exclusion) — baseline
  2. Exclude l=3 only
  3. Exclude l=3, 7, 9 (the primary triad)
  4. Exclude l=3, 7, 9, 18, 63 (all individually significant at <2%)
  5. Exclude ALL l <= 15 (remove entire low-l region)
  6. ONLY l <= 30 vs ONLY l > 30 (split test)

For each, reports: M matrix, SVD, rank-1%, u, v, and comparison to full.

Usage:
  python d3_factorization_robustness.py --results extended_results --lmax 150
"""

import numpy as np
import json, os, argparse, glob, time


def load_batch1_results(results_dir):
    maps = {}
    for name in ['smica', 'nilc', 'sevem', 'commander']:
        pattern = os.path.join(results_dir, f'd3_extended_{name}_lmax*.json')
        matches = glob.glob(pattern)
        if matches:
            with open(matches[0]) as f:
                maps[name.upper()] = json.load(f)
    return maps


def expected_fractions(l):
    n_a2 = l // 3
    dof_A1 = 1 + n_a2
    dof_A2 = n_a2
    dof_E = 2 * (l - n_a2)
    total = 2 * l + 1
    return dof_A1 / total, dof_A2 / total, dof_E / total


def build_transfer_matrix(per_l, lmax, exclude_ls=None, include_range=None):
    """Build 2×3 transfer matrix with optional exclusions.

    exclude_ls: set of l values to skip
    include_range: (lmin, lmax) tuple — only include l in this range
    """
    if exclude_ls is None:
        exclude_ls = set()

    M = np.zeros((2, 3))  # rows: odd=0, even=1; cols: r=0,1,2

    for l in range(2, lmax + 1):
        if l in exclude_ls:
            continue
        if include_range is not None:
            if l < include_range[0] or l > include_range[1]:
                continue

        l_str = str(l)
        if l_str not in per_l:
            continue

        d = per_l[l_str]
        delta_A2 = d['delta_A2']
        row = 0 if l % 2 == 1 else 1  # odd=0, even=1
        col = l % 3
        M[row, col] += delta_A2

    return M


def analyze_matrix(M, label=""):
    """SVD analysis of a 2×3 matrix. Returns dict with results."""
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    total_var = np.sum(s**2)
    rank1_frac = s[0]**2 / total_var if total_var > 0 else 0

    u = U[:, 0].copy()
    v = Vt[0, :].copy()

    # Sign-normalize: odd (row 0) should be positive
    if u[0] < 0:
        u = -u
        v = -v

    return {
        'label': label,
        'M': M.copy(),
        's': s.copy(),
        'u': u,
        'v': v,
        'rank1_frac': float(rank1_frac),
        'sigma_ratio': float(s[0] / s[1]) if s[1] > 1e-15 else float('inf'),
    }


def print_comparison(results, baseline_key='full'):
    """Print comparison table across exclusion sets."""
    baseline = results[baseline_key]

    print(f"\n  {'Exclusion':>30} {'Rank-1%':>8} {'sig1':>8} {'sig2':>8} "
          f"{'u_odd':>7} {'u_even':>7} {'v_r0':>7} {'v_r1':>7} {'v_r2':>7} "
          f"{'cos(u)':>7} {'cos(v)':>7}")
    print(f"  {'-'*108}")

    for key, r in results.items():
        cos_u = np.dot(r['u'], baseline['u'])
        cos_v = np.dot(r['v'], baseline['v'])

        is_base = " *" if key == baseline_key else ""
        print(f"  {r['label']:>30} {r['rank1_frac']*100:>7.1f}% {r['s'][0]:>8.4f} {r['s'][1]:>8.4f} "
              f"{r['u'][0]:>+7.3f} {r['u'][1]:>+7.3f} "
              f"{r['v'][0]:>+7.3f} {r['v'][1]:>+7.3f} {r['v'][2]:>+7.3f} "
              f"{cos_u:>+7.3f} {cos_v:>+7.3f}{is_base}")


def run(results_dir, lmax=150):
    t0 = time.time()
    maps_data = load_batch1_results(results_dir)

    if not maps_data:
        print("ERROR: No Batch 1 result files found")
        return

    print(f"{'#'*72}")
    print(f"# 2×3 FACTORIZATION ROBUSTNESS TEST")
    print(f"# Maps: {', '.join(maps_data.keys())}")
    print(f"# lmax: {lmax}")
    print(f"# Key question: does factorization survive without l=3,7,9?")
    print(f"{'#'*72}")

    # Define exclusion sets
    exclusion_sets = {
        'full': {'label': 'Full (no exclusion)', 'exclude': set()},
        'no_3': {'label': 'Exclude l=3', 'exclude': {3}},
        'no_3_7_9': {'label': 'Exclude l=3,7,9', 'exclude': {3, 7, 9}},
        'no_top5': {'label': 'Exclude l=3,7,9,18,63', 'exclude': {3, 7, 9, 18, 63}},
        'no_low15': {'label': 'Exclude all l<=15', 'exclude': set(range(2, 16))},
        'no_low30': {'label': 'Exclude all l<=30', 'exclude': set(range(2, 31))},
    }

    # Range-based splits
    range_sets = {
        'only_low30': {'label': 'Only l=2..30', 'range': (2, 30)},
        'only_high': {'label': 'Only l=31..150', 'range': (31, 150)},
        'only_mid': {'label': 'Only l=16..80', 'range': (16, 80)},
        'only_high80': {'label': 'Only l=81..150', 'range': (81, 150)},
    }

    all_output = {}

    for map_label, data in maps_data.items():
        per_l = data['per_l']

        print(f"\n{'='*72}")
        print(f"  MAP: {map_label}")
        print(f"{'='*72}")

        results = {}

        # Exclusion-based tests
        for key, spec in exclusion_sets.items():
            M = build_transfer_matrix(per_l, lmax, exclude_ls=spec['exclude'])
            results[key] = analyze_matrix(M, spec['label'])

        # Range-based tests
        for key, spec in range_sets.items():
            M = build_transfer_matrix(per_l, lmax, include_range=spec['range'])
            results[key] = analyze_matrix(M, spec['label'])

        print_comparison(results, baseline_key='full')

        # Print the actual matrices for the key comparison
        print(f"\n  Transfer matrices:")
        for key in ['full', 'no_3_7_9', 'no_low15', 'only_high']:
            r = results[key]
            M = r['M']
            print(f"\n  {r['label']}:")
            print(f"  {'':>8} {'r=0':>10} {'r=1':>10} {'r=2':>10}")
            print(f"  {'odd':>8} {M[0,0]:>+10.4f} {M[0,1]:>+10.4f} {M[0,2]:>+10.4f}")
            print(f"  {'even':>8} {M[1,0]:>+10.4f} {M[1,1]:>+10.4f} {M[1,2]:>+10.4f}")

        # Store for JSON
        all_output[map_label] = {}
        for key, r in results.items():
            all_output[map_label][key] = {
                'label': r['label'],
                'rank1_frac': r['rank1_frac'],
                'sigma1': float(r['s'][0]),
                'sigma2': float(r['s'][1]),
                'u': r['u'].tolist(),
                'v': r['v'].tolist(),
                'M': r['M'].tolist(),
                'cos_u_vs_full': float(np.dot(r['u'], results['full']['u'])),
                'cos_v_vs_full': float(np.dot(r['v'], results['full']['v'])),
            }

    # ================================================================
    # CROSS-MAP COMPARISON FOR KEY EXCLUSION: no l=3,7,9
    # ================================================================
    print(f"\n{'='*72}")
    print("CROSS-MAP COMPARISON: EXCLUDING l=3,7,9")
    print(f"{'='*72}")

    labels = list(maps_data.keys())

    # Collect u and v vectors for no_3_7_9
    u_vecs = {}
    v_vecs = {}
    for map_label, data in maps_data.items():
        per_l = data['per_l']
        M = build_transfer_matrix(per_l, lmax, exclude_ls={3, 7, 9})
        r = analyze_matrix(M)
        u_vecs[map_label] = r['u']
        v_vecs[map_label] = r['v']

    print(f"\n  u vectors (excluding l=3,7,9):")
    for lab in labels:
        u = u_vecs[lab]
        print(f"    {lab:>12}: ({u[0]:+.4f}, {u[1]:+.4f})")

    print(f"\n  v vectors (excluding l=3,7,9):")
    for lab in labels:
        v = v_vecs[lab]
        print(f"    {lab:>12}: ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")

    print(f"\n  Cross-map u cosine (excl l=3,7,9):")
    print(f"  {'':>12}", end="")
    for lab in labels:
        print(f" {lab:>12}", end="")
    print()
    for l1 in labels:
        print(f"  {l1:>12}", end="")
        for l2 in labels:
            print(f" {np.dot(u_vecs[l1], u_vecs[l2]):>+12.4f}", end="")
        print()

    print(f"\n  Cross-map v cosine (excl l=3,7,9):")
    print(f"  {'':>12}", end="")
    for lab in labels:
        print(f" {lab:>12}", end="")
    print()
    for l1 in labels:
        print(f"  {l1:>12}", end="")
        for l2 in labels:
            print(f" {np.dot(v_vecs[l1], v_vecs[l2]):>+12.4f}", end="")
        print()

    # ================================================================
    # CROSS-MAP: EXCLUDING ALL l<=15
    # ================================================================
    print(f"\n{'='*72}")
    print("CROSS-MAP COMPARISON: EXCLUDING ALL l<=15")
    print(f"{'='*72}")

    u_vecs2 = {}
    v_vecs2 = {}
    for map_label, data in maps_data.items():
        per_l = data['per_l']
        M = build_transfer_matrix(per_l, lmax, exclude_ls=set(range(2, 16)))
        r = analyze_matrix(M)
        u_vecs2[map_label] = r['u']
        v_vecs2[map_label] = r['v']

    print(f"\n  u vectors (l=16..150 only):")
    for lab in labels:
        u = u_vecs2[lab]
        print(f"    {lab:>12}: ({u[0]:+.4f}, {u[1]:+.4f})")

    print(f"\n  v vectors (l=16..150 only):")
    for lab in labels:
        v = v_vecs2[lab]
        print(f"    {lab:>12}: ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")

    print(f"\n  Cross-map v cosine (l=16..150):")
    print(f"  {'':>12}", end="")
    for lab in labels:
        print(f" {lab:>12}", end="")
    print()
    for l1 in labels:
        print(f"  {l1:>12}", end="")
        for l2 in labels:
            print(f" {np.dot(v_vecs2[l1], v_vecs2[l2]):>+12.4f}", end="")
        print()

    # ================================================================
    # MASTER VERDICT
    # ================================================================
    print(f"\n{'='*72}")
    print("MASTER VERDICT: ROBUSTNESS OF FACTORIZATION")
    print(f"{'='*72}")

    print(f"\n  Does the factorization survive without l=3,7,9?")
    print(f"\n  {'Map':>12} {'Full rank1%':>12} {'No 3,7,9':>12} {'No l<=15':>12} {'l>30 only':>12}")
    print(f"  {'-'*60}")

    for map_label, data in maps_data.items():
        per_l = data['per_l']

        M_full = build_transfer_matrix(per_l, lmax)
        M_no379 = build_transfer_matrix(per_l, lmax, exclude_ls={3, 7, 9})
        M_no15 = build_transfer_matrix(per_l, lmax, exclude_ls=set(range(2, 16)))
        M_high = build_transfer_matrix(per_l, lmax, include_range=(31, 150))

        r_full = analyze_matrix(M_full)
        r_no379 = analyze_matrix(M_no379)
        r_no15 = analyze_matrix(M_no15)
        r_high = analyze_matrix(M_high)

        print(f"  {map_label:>12} {r_full['rank1_frac']*100:>11.1f}% "
              f"{r_no379['rank1_frac']*100:>11.1f}% "
              f"{r_no15['rank1_frac']*100:>11.1f}% "
              f"{r_high['rank1_frac']*100:>11.1f}%")

    # Check u/v survival
    print(f"\n  u-vector structure (odd+, even-) survives?")
    for map_label, data in maps_data.items():
        per_l = data['per_l']
        M = build_transfer_matrix(per_l, lmax, exclude_ls={3, 7, 9})
        r = analyze_matrix(M)
        survives = r['u'][0] > 0 and r['u'][1] < 0
        print(f"    {map_label:>12}: u=({r['u'][0]:+.3f}, {r['u'][1]:+.3f}) → {'YES' if survives else 'NO'}")

    print(f"\n  v-vector structure (r=0 dominant) survives?")
    for map_label, data in maps_data.items():
        per_l = data['per_l']
        M = build_transfer_matrix(per_l, lmax, exclude_ls={3, 7, 9})
        r = analyze_matrix(M)
        dominant = r['v'][0] > abs(r['v'][1]) and r['v'][0] > abs(r['v'][2])
        print(f"    {map_label:>12}: v=({r['v'][0]:+.3f}, {r['v'][1]:+.3f}, {r['v'][2]:+.3f}) → {'YES' if dominant else 'NO'}")

    # Final answer
    print(f"\n  {'='*50}")

    all_survive_u = True
    all_survive_v = True
    all_survive_rank = True
    for map_label, data in maps_data.items():
        per_l = data['per_l']
        M = build_transfer_matrix(per_l, lmax, exclude_ls={3, 7, 9})
        r = analyze_matrix(M)
        if not (r['u'][0] > 0 and r['u'][1] < 0):
            all_survive_u = False
        if not (r['v'][0] > abs(r['v'][1]) and r['v'][0] > abs(r['v'][2])):
            all_survive_v = False
        if r['rank1_frac'] < 0.70:
            all_survive_rank = False

    if all_survive_u and all_survive_v and all_survive_rank:
        print(f"  ANSWER: YES — factorization survives removal of l=3,7,9")
        print(f"  The 2×3 grammar is NOT a low-l curiosity.")
    else:
        print(f"  ANSWER: PARTIAL — some structure survives, some degrades")
        print(f"    u survives: {all_survive_u}")
        print(f"    v survives: {all_survive_v}")
        print(f"    rank1 > 70%: {all_survive_rank}")

    print(f"  {'='*50}")

    # Save
    outfile = os.path.join(results_dir, 'factorization_robustness.json')
    with open(outfile, 'w') as f:
        json.dump(all_output, f, indent=2)
    print(f"\n  Results saved to {outfile}")
    print(f"  Total time: {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='2×3 Factorization Robustness Test')
    parser.add_argument('--results', required=True, help='Directory with Batch 1 JSONs')
    parser.add_argument('--lmax', type=int, default=150)
    args = parser.parse_args()
    run(args.results, args.lmax)


if __name__ == '__main__':
    main()
