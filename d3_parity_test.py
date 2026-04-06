#!/usr/bin/env python3
"""
D3 odd/even parity test.

Tests whether the D3 A2-excess signal differs between odd-l and even-l multipoles.

D3 = S3 has no parity element (all elements are proper rotations + reflections in
the plane, not inversion). Therefore the IPT boundary-filter prediction is:
  - No systematic odd/even asymmetry in f_A2 excess.

A parity-dependent signal would indicate contamination from a parity-violating
source (e.g., foreground residuals, systematics).

Tests performed:
  1. Mean delta_A2 for odd vs even l (with permutation test)
  2. Fisher combined PTE separately for odd and even l
  3. Cross-map consistency of the odd/even split

Usage:
  python d3_parity_test.py --results dir/ [--lmax 150]
"""

import numpy as np
import json, os, argparse, glob
from scipy.stats import chi2


def load_results(results_dir):
    """Load all per-map extended result JSONs."""
    maps = {}
    for name in ['smica', 'nilc', 'sevem', 'commander']:
        pattern = os.path.join(results_dir, f'd3_extended_{name}_lmax*.json')
        matches = glob.glob(pattern)
        if matches:
            with open(matches[0]) as f:
                maps[name.upper()] = json.load(f)
    return maps


def parity_analysis(data, label, lmax=150):
    """Run parity analysis on a single map's results."""
    per_l = data['per_l']

    odd_delta, even_delta = [], []
    odd_pte, even_pte = [], []
    odd_fA2, even_fA2 = [], []

    for l_str, d in per_l.items():
        l = int(l_str)
        if l < 2 or l > lmax:
            continue

        delta = d['delta_A2']
        pte = d['pte_A2']
        fA2 = d['f_A2']

        if l % 2 == 1:  # odd
            odd_delta.append(delta)
            odd_fA2.append(fA2)
            if pte is not None and not np.isnan(pte):
                odd_pte.append(pte)
        else:  # even
            even_delta.append(delta)
            even_fA2.append(fA2)
            if pte is not None and not np.isnan(pte):
                even_pte.append(pte)

    odd_delta = np.array(odd_delta)
    even_delta = np.array(even_delta)

    # Mean excess
    mean_odd = np.mean(odd_delta)
    mean_even = np.mean(even_delta)
    diff = mean_odd - mean_even

    # Permutation test for odd-even difference
    n_perm = 100000
    rng = np.random.default_rng(42)
    all_delta = np.concatenate([odd_delta, even_delta])
    n_odd = len(odd_delta)
    perm_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(all_delta)
        perm_diffs[i] = np.mean(perm[:n_odd]) - np.mean(perm[n_odd:])

    perm_pte = float(np.mean(np.abs(perm_diffs) >= np.abs(diff)))

    # Fisher combined PTE for odd and even separately
    fisher_odd = -2 * sum(np.log(max(p, 1e-15)) for p in odd_pte)
    fisher_even = -2 * sum(np.log(max(p, 1e-15)) for p in even_pte)
    dof_odd = 2 * len(odd_pte)
    dof_even = 2 * len(even_pte)
    fisher_p_odd = float(chi2.sf(fisher_odd, dof_odd))
    fisher_p_even = float(chi2.sf(fisher_even, dof_even))

    # Count significant multipoles
    n_sig_odd = sum(1 for p in odd_pte if p < 0.05)
    n_sig_even = sum(1 for p in even_pte if p < 0.05)

    # l mod 3 breakdown within odd/even
    odd_mod3_0 = [d for l_str, d in per_l.items()
                  if int(l_str) % 2 == 1 and int(l_str) % 3 == 0 and 2 <= int(l_str) <= lmax]
    even_mod3_0 = [d for l_str, d in per_l.items()
                   if int(l_str) % 2 == 0 and int(l_str) % 3 == 0 and 2 <= int(l_str) <= lmax]
    mean_odd_mod3 = np.mean([d['delta_A2'] for d in odd_mod3_0]) if odd_mod3_0 else np.nan
    mean_even_mod3 = np.mean([d['delta_A2'] for d in even_mod3_0]) if even_mod3_0 else np.nan

    result = {
        'n_odd': len(odd_delta),
        'n_even': len(even_delta),
        'mean_delta_A2_odd': float(mean_odd),
        'mean_delta_A2_even': float(mean_even),
        'diff_odd_minus_even': float(diff),
        'perm_pte_two_sided': perm_pte,
        'fisher_odd': float(fisher_odd),
        'fisher_p_odd': fisher_p_odd,
        'fisher_dof_odd': dof_odd,
        'fisher_even': float(fisher_even),
        'fisher_p_even': fisher_p_even,
        'fisher_dof_even': dof_even,
        'n_sig_005_odd': n_sig_odd,
        'n_sig_005_even': n_sig_even,
        'mean_fA2_odd': float(np.mean(odd_fA2)),
        'mean_fA2_even': float(np.mean(even_fA2)),
        'mean_delta_A2_odd_mod3_0': float(mean_odd_mod3),
        'mean_delta_A2_even_mod3_0': float(mean_even_mod3),
    }

    return result


def run(results_dir, lmax=150):
    maps = load_results(results_dir)
    if not maps:
        print(f"ERROR: No result files found in {results_dir}")
        return

    all_results = {}

    print(f"{'='*72}")
    print("D3 ODD/EVEN PARITY TEST")
    print(f"  lmax={lmax}, N_perm=100,000")
    print(f"  Maps: {', '.join(maps.keys())}")
    print(f"{'='*72}")

    for label, data in maps.items():
        print(f"\n--- {label} ---")
        r = parity_analysis(data, label, lmax)
        all_results[label] = r

        print(f"  Odd l:  n={r['n_odd']}, mean dA2={r['mean_delta_A2_odd']:+.5f}, "
              f"Fisher p={r['fisher_p_odd']:.4f}, sig@5%={r['n_sig_005_odd']}")
        print(f"  Even l: n={r['n_even']}, mean dA2={r['mean_delta_A2_even']:+.5f}, "
              f"Fisher p={r['fisher_p_even']:.4f}, sig@5%={r['n_sig_005_even']}")
        print(f"  Diff (odd-even): {r['diff_odd_minus_even']:+.5f}, "
              f"perm PTE={r['perm_pte_two_sided']:.4f}")
        print(f"  l mod 3==0 subset: odd={r['mean_delta_A2_odd_mod3_0']:+.5f}, "
              f"even={r['mean_delta_A2_even_mod3_0']:+.5f}")

        if r['perm_pte_two_sided'] < 0.05:
            print(f"  ** PARITY ASYMMETRY DETECTED (p={r['perm_pte_two_sided']:.4f})")
        else:
            print(f"  Null confirmed: no parity asymmetry")

    # Cross-map summary
    print(f"\n{'='*72}")
    print("CROSS-MAP PARITY SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Map':>12} {'dA2(odd)':>10} {'dA2(even)':>10} {'diff':>10} "
          f"{'perm PTE':>10} {'Fisher(odd)':>12} {'Fisher(even)':>12}")
    print(f"  {'-'*78}")
    for lab, r in all_results.items():
        print(f"  {lab:>12} {r['mean_delta_A2_odd']:>+10.5f} {r['mean_delta_A2_even']:>+10.5f} "
              f"{r['diff_odd_minus_even']:>+10.5f} {r['perm_pte_two_sided']:>10.4f} "
              f"{r['fisher_p_odd']:>12.4f} {r['fisher_p_even']:>12.4f}")

    # Check cross-map consistency of parity difference sign
    diffs = [r['diff_odd_minus_even'] for r in all_results.values()]
    same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    print(f"\n  All maps same sign? {'YES' if same_sign else 'NO'} "
          f"(diffs: {[f'{d:+.5f}' for d in diffs]})")

    outfile = os.path.join(results_dir, 'parity_test.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")


def main():
    parser = argparse.ArgumentParser(description='D3 Odd/Even Parity Test')
    parser.add_argument('--results', required=True, help='Directory with extended result JSONs')
    parser.add_argument('--lmax', type=int, default=150)
    args = parser.parse_args()
    run(args.results, args.lmax)


if __name__ == '__main__':
    main()
