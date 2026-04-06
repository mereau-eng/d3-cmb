#!/usr/bin/env python3
"""
D₃ symmetry analysis of Planck E-mode polarization.
Uses pre-extracted alms from planck_ee_alms.json.
"""

import json, numpy as np, time
from scipy.spatial.transform import Rotation as R
from scipy.special import factorial
from scipy.optimize import differential_evolution

t0 = time.time()

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("D₃ SYMMETRY IN PLANCK E-MODE POLARIZATION")
print("=" * 70)

with open('/sessions/trusting-affectionate-turing/planck_ee_alms.json') as f:
    pol_data = json.load(f)

with open('/sessions/trusting-affectionate-turing/mnt/IPT_Cline/IPT_CMB/results/planck_PR3_2018_SMICA_fullsky_alm_lmax10.json') as f:
    tt_data = json.load(f)

with open('/sessions/trusting-affectionate-turing/mnt/IPT_Cline/IPT_CMB/results/optimal_d3_axis.json') as f:
    axis_data = json.load(f)

opt_l = axis_data['optimal_axis']['l']
opt_b = axis_data['optimal_axis']['b']
print(f"TT-optimal D₃ axis: (l, b) = ({opt_l:.1f}°, {opt_b:.1f}°)")

ee_dict = pol_data['E']
tt_dict_hp = pol_data['T']  # T from same healpy extraction for cross-check

# ============================================================
# CORE D₃ FUNCTIONS
# ============================================================
def galactic_to_cart(l_deg, b_deg):
    lr, br = np.radians(l_deg), np.radians(b_deg)
    return np.array([np.cos(br)*np.cos(lr), np.cos(br)*np.sin(lr), np.sin(br)])

def wigner_d_element(l, m, mp, beta):
    c, s = np.cos(beta/2), np.sin(beta/2)
    d_sum = 0.0
    for k in range(max(0, mp-m), min(l+mp, l-m)+1):
        term = ((-1)**(m-mp+k) * c**(2*l+mp-m-2*k) * s**(m-mp+2*k))
        denom = factorial(l+mp-k)*factorial(k)*factorial(m-mp+k)*factorial(l-m-k)
        d_sum += term / denom
    return float(np.sqrt(factorial(l+m)*factorial(l-m)*factorial(l+mp)*factorial(l-mp)) * d_sum)

def wigner_d_matrix(l, alpha, beta, gamma):
    size = 2*l+1
    D = np.zeros((size, size), dtype=complex)
    for mi, m in enumerate(range(-l, l+1)):
        for mpi, mp in enumerate(range(-l, l+1)):
            D[mi, mpi] = np.exp(-1j*m*alpha) * wigner_d_element(l, m, mp, beta) * np.exp(-1j*mp*gamma)
    return D

def axis_to_euler(axis):
    axis = axis / np.linalg.norm(axis)
    z = np.array([0., 0., 1.])
    if abs(np.dot(axis, z) - 1.0) < 1e-10:
        return (0., 0., 0.)
    if abs(np.dot(axis, z) + 1.0) < 1e-10:
        rot = R.from_euler('x', 180, degrees=True)
        return tuple(rot.as_euler('ZYZ', degrees=False))
    rot_ax = np.cross(axis, z); rot_ax /= np.linalg.norm(rot_ax)
    angle = np.arccos(np.clip(np.dot(axis, z), -1, 1))
    return tuple(R.from_rotvec(angle * rot_ax).as_euler('ZYZ', degrees=False))

def get_alm_vec(alm_dict, l):
    return np.array([alm_dict[str(m)][0] + 1j*alm_dict[str(m)][1] for m in range(-l, l+1)])

# Pre-compute projectors
projectors = {}
for l in range(2, 11):
    size = 2*l+1
    I = np.eye(size, dtype=complex)
    r = np.diag([np.exp(2j*np.pi*m/3) for m in range(-l, l+1)])
    r2 = r @ r
    sig = np.zeros((size, size), dtype=complex)
    for mi, m in enumerate(range(-l, l+1)):
        sig[(-m)-(-l), mi] = (-1)**m
    projectors[l] = (
        (I + r + r2 + sig + sig@r + sig@r2) / 6,  # A1
        (I + r + r2 - sig - sig@r - sig@r2) / 6,  # A2
        (2*I - r - r2) / 3                          # E
    )

def irrep_fracs(vec, l):
    P1, P2, PE = projectors[l]
    tot = np.sum(np.abs(vec)**2)
    if tot == 0: return 0., 0., 0.
    return (np.sum(np.abs(P1@vec)**2)/tot, np.sum(np.abs(P2@vec)**2)/tot, np.sum(np.abs(PE@vec)**2)/tot)

# ============================================================
# TT vs EE COMPARISON AT TT-OPTIMAL AXIS
# ============================================================
print("\n" + "=" * 70)
print("D₃ IRREP DECOMPOSITION: TT vs EE (at TT-optimal axis)")
print("=" * 70)

opt_axis = galactic_to_cart(opt_l, opt_b)
opt_euler = axis_to_euler(opt_axis)

# Pre-compute D matrices for this axis
Dmats = {l: wigner_d_matrix(l, *opt_euler) for l in range(2, 11)}

print(f"\n{'l':>3} | {'f_A2(TT)':>10} {'f_A1(TT)':>10} {'f_E(TT)':>10} | {'f_A2(EE)':>10} {'f_A1(EE)':>10} {'f_E(EE)':>10}")
print("-" * 82)

tt_fA2 = {}; ee_fA2 = {}

for l in range(2, 11):
    # TT — use original pre-extracted alms from the paper analysis
    tt_vec = get_alm_vec(tt_data['alm'][str(l)], l)
    tt_rot = Dmats[l] @ tt_vec
    tA1, tA2, tE = irrep_fracs(tt_rot, l)
    tt_fA2[l] = tA2

    # EE
    ee_vec = get_alm_vec(ee_dict[str(l)], l)
    ee_rot = Dmats[l] @ ee_vec
    eA1, eA2, eE = irrep_fracs(ee_rot, l)
    ee_fA2[l] = eA2

    ee_mark = " ◄" if eA2 > 0.50 else ""
    print(f"{l:>3} | {tA2:>10.4f} {tA1:>10.4f} {tE:>10.4f} | {eA2:>10.4f} {eA1:>10.4f} {eE:>10.4f}{ee_mark}")


# ============================================================
# MONTE CARLO PTE
# ============================================================
N_MC = 5000
rng = np.random.default_rng(42)
print(f"\n{'='*70}")
print(f"MONTE CARLO PTE (N={N_MC}, fixed TT axis, CS-constrained)")
print(f"{'='*70}")

mc_fA2 = {l: np.zeros(N_MC) for l in range(2, 11)}

for i in range(N_MC):
    for l in range(2, 11):
        alm = np.zeros(2*l+1, dtype=complex)
        alm[l] = rng.standard_normal()
        for m in range(1, l+1):
            re, im = rng.standard_normal(), rng.standard_normal()
            alm[l+m] = re + 1j*im
            alm[l-m] = ((-1)**m)*(re - 1j*im)
        _, fA2, _ = irrep_fracs(alm, l)
        mc_fA2[l][i] = fA2

print(f"\n{'l':>3} | {'f_A2(EE)':>10} | {'MC mean':>10} {'MC std':>10} | {'PTE':>10} | {'z':>8}")
print("-" * 70)

ee_ptes = {}
for l in range(2, 11):
    obs = ee_fA2[l]
    mc = mc_fA2[l]
    mn, sd = np.mean(mc), np.std(mc)
    pte = np.mean(mc >= obs)
    z = (obs - mn)/sd if sd > 0 else 0
    ee_ptes[l] = pte
    flag = " ***" if pte < 0.001 else (" **" if pte < 0.01 else (" *" if pte < 0.05 else ""))
    pte_s = f"{pte:.4f}" if pte > 0 else f"< {1/N_MC:.1e}"
    print(f"{l:>3} | {obs:>10.4f} | {mn:>10.4f} {sd:>10.4f} | {pte_s:>10} | {z:>+8.2f}{flag}")


# ============================================================
# EE AXIS OPTIMIZATION
# ============================================================
print(f"\n{'='*70}")
print("EE-INDEPENDENT AXIS OPTIMIZATION")
print(f"{'='*70}")

def neg_fA2(params, l_val, alm_src):
    axis = galactic_to_cart(*params)
    euler = axis_to_euler(axis)
    D = wigner_d_matrix(l_val, *euler)
    rot = D @ get_alm_vec(alm_src[str(l_val)], l_val)
    _, fA2, _ = irrep_fracs(rot, l_val)
    return -fA2

# l=3
res3 = differential_evolution(neg_fA2, [(0,360),(-90,90)], args=(3, ee_dict), seed=42, maxiter=200)
ee_opt3 = res3.x
sep3 = np.degrees(np.arccos(np.clip(abs(np.dot(opt_axis, galactic_to_cart(*ee_opt3))), -1, 1)))
print(f"\nEE-optimal (l=3): ({ee_opt3[0]:.1f}°, {ee_opt3[1]:.1f}°), f_A2={-res3.fun:.4f}, sep={sep3:.1f}°")

# Multi-l
def neg_mean_fA2(params, l_list, alm_src):
    axis = galactic_to_cart(*params)
    euler = axis_to_euler(axis)
    s = 0
    for lv in l_list:
        D = wigner_d_matrix(lv, *euler)
        rot = D @ get_alm_vec(alm_src[str(lv)], lv)
        _, fA2, _ = irrep_fracs(rot, lv)
        s += fA2
    return -s/len(l_list)

key_ls = [3, 6, 7, 9]
resm = differential_evolution(neg_mean_fA2, [(0,360),(-90,90)], args=(key_ls, ee_dict), seed=42, maxiter=200)
ee_optm = resm.x
sepm = np.degrees(np.arccos(np.clip(abs(np.dot(opt_axis, galactic_to_cart(*ee_optm))), -1, 1)))
print(f"EE-optimal (multi): ({ee_optm[0]:.1f}°, {ee_optm[1]:.1f}°), mean f_A2={-resm.fun:.4f}, sep={sepm:.1f}°")

# All l=2..10 optimization
all_ls = list(range(2, 11))
resa = differential_evolution(neg_mean_fA2, [(0,360),(-90,90)], args=(all_ls, ee_dict), seed=42, maxiter=200)
ee_opta = resa.x
sepa = np.degrees(np.arccos(np.clip(abs(np.dot(opt_axis, galactic_to_cart(*ee_opta))), -1, 1)))
print(f"EE-optimal (all l): ({ee_opta[0]:.1f}°, {ee_opta[1]:.1f}°), mean f_A2={-resa.fun:.4f}, sep={sepa:.1f}°")

# Decomposition at EE-optimal multi-l axis
print(f"\nFull decomposition at EE-optimal multi-l axis ({ee_optm[0]:.1f}°, {ee_optm[1]:.1f}°):")
print(f"{'l':>3} | {'f_A2':>10} {'f_A1':>10} {'f_E':>10}")
print("-" * 42)
ee_opt_euler = axis_to_euler(galactic_to_cart(*ee_optm))
for l in range(2, 11):
    D = wigner_d_matrix(l, *ee_opt_euler)
    rot = D @ get_alm_vec(ee_dict[str(l)], l)
    fA1, fA2, fE = irrep_fracs(rot, l)
    mark = " ◄" if fA2 > 0.50 else ""
    print(f"{l:>3} | {fA2:>10.4f} {fA1:>10.4f} {fE:>10.4f}{mark}")


# ============================================================
# TT-EE CORRELATION
# ============================================================
print(f"\n{'='*70}")
print("TT-EE CORRELATION (at TT-optimal axis)")
print(f"{'='*70}")

tt_v = np.array([tt_fA2[l] for l in range(2, 11)])
ee_v = np.array([ee_fA2[l] for l in range(2, 11)])
corr = np.corrcoef(tt_v, ee_v)[0, 1]
print(f"\nPearson r = {corr:.4f}")
print(f"\n{'l':>3} | {'TT f_A2':>10} | {'EE f_A2':>10} | {'both>0.5?':>10}")
print("-" * 50)
for l in range(2, 11):
    both = "YES" if tt_fA2[l] > 0.5 and ee_fA2[l] > 0.5 else ""
    print(f"{l:>3} | {tt_fA2[l]:>10.4f} | {ee_fA2[l]:>10.4f} | {both:>10}")

# Fisher's method on EE PTEs for key multipoles
from scipy.stats import chi2
key_ptes = [ee_ptes[l] for l in key_ls if ee_ptes[l] > 0]
if key_ptes:
    fisher_stat = -2 * sum(np.log(p) for p in key_ptes)
    fisher_p = chi2.sf(fisher_stat, 2*len(key_ptes))
    print(f"\nFisher combined PTE for l={key_ls}: {fisher_p:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

n_excess = sum(1 for l in range(2,11) if ee_fA2[l] > 0.5)
n_sig = sum(1 for l in range(2,11) if ee_ptes[l] < 0.05)

print(f"""
Results at TT-optimal axis:
  EE multipoles with f_A2 > 50%: {n_excess}/9
  EE multipoles with PTE < 0.05: {n_sig}/9
  TT-EE f_A2 correlation: r = {corr:.4f}

Axis alignment:
  TT-optimal:          ({opt_l:.1f}°, {opt_b:.1f}°)
  EE-optimal (l=3):    ({ee_opt3[0]:.1f}°, {ee_opt3[1]:.1f}°)  [{sep3:.0f}° away]
  EE-optimal (multi):  ({ee_optm[0]:.1f}°, {ee_optm[1]:.1f}°)  [{sepm:.0f}° away]
  EE-optimal (all l):  ({ee_opta[0]:.1f}°, {ee_opta[1]:.1f}°)  [{sepa:.0f}° away]

Time: {time.time()-t0:.1f}s
""")
