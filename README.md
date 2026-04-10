# D3 = Weyl(SU(3)) Symmetry in the CMB

**Companion code for:**
*D3 = Weyl(SU(3)) Discrete Symmetry in the Cosmic Microwave Background*
Robert Mereau (2026)

This repository contains all analysis scripts that produce the figures, tables, and statistical tests reported in the paper. Each script is self-contained and operates on publicly available Planck PR3 and WMAP 9-year data products.

---

## Repository Structure

```
d3-cmb/
  scripts/                              # Core analysis (main paper)
    d3_extended_analysis.py             # Tables 1-5, 9-10; Figures 1-6, 9
    d3_polarization_analysis.py         # EE null test (Section 6.2)
    d3_factorization_robustness.py      # Rank-1 factorization (Table 8)
    d3_parity_test.py                   # Parity gate (Table 6)
    d3_phase_locking_test.py            # Phase-locking null test
    d3_witness_transfer_analysis.py     # Transfer grammar (Table 7, Figure 10)
  wmap_d3/scripts/                      # WMAP replication (Appendix B)
    wmap_d3_analysis.py
  other_groups/scripts/                 # Group specificity (Appendix C)
    group_comparison_analysis.py
  freq_channels/scripts/                # Frequency channels (Appendix D)
    freq_channel_d3_analysis.py
  mask_variation/scripts/               # Mask variation (Appendix E)
    mask_variation_analysis.py
```

## Quick Start

### Requirements

```
Python >= 3.9
numpy >= 1.24
scipy >= 1.10
healpy >= 1.16
```

Install:
```bash
pip install numpy scipy healpy
```

### Data

All analyses use publicly available data:

- **Planck PR3**: Component-separated CMB maps (SMICA, NILC, Commander, SEVEM) from [IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/) or the [Planck Legacy Archive](https://pla.esac.esa.int/)
- **WMAP 9-year**: ILC, V-band, and W-band maps from [LAMBDA](https://lambda.gsfc.nasa.gov/product/wmap/current/)
- **Planck HFI frequency maps**: 100, 143, 217, 353 GHz from IRSA
- **Masks**: UT78 common mask, GAL series, point source masks from IRSA

### Running the Core Analysis

```bash
# Main D3 decomposition (Tables 1-5, Figures 1-6)
python scripts/d3_extended_analysis.py \
  --fits COM_CMB_IQU-smica_2048_R3.00_full.fits \
  --lmax 150 --nsims 10000 --outdir results/

# Parity gate (Table 6)
python scripts/d3_parity_test.py \
  --fits COM_CMB_IQU-smica_2048_R3.00_full.fits \
  --outdir results/ --nsims 10000

# Transfer grammar (Table 7, Figure 10)
python scripts/d3_witness_transfer_analysis.py \
  --smica COM_CMB_IQU-smica_2048_R3.00_full.fits \
  --nilc COM_CMB_IQU-nilc_2048_R3.00_full.fits \
  --commander COM_CMB_IQU-commander_2048_R3.00_full.fits \
  --sevem COM_CMB_IQU-sevem_2048_R3.00_full.fits \
  --outdir results/ --nsims 10000

# Rank-1 factorization (Table 8)
python scripts/d3_factorization_robustness.py \
  --smica COM_CMB_IQU-smica_2048_R3.00_full.fits \
  --nilc COM_CMB_IQU-nilc_2048_R3.00_full.fits \
  --commander COM_CMB_IQU-commander_2048_R3.00_full.fits \
  --sevem COM_CMB_IQU-sevem_2048_R3.00_full.fits \
  --outdir results/
```

### Running the Appendix Analyses

```bash
# Appendix B: WMAP 9-year replication
python wmap_d3/scripts/wmap_d3_analysis.py \
  --datadir wmap_d3/data/ --outdir wmap_d3/results/ --nmc 10000

# Appendix C: D3 vs D4, D5, D6 specificity
python other_groups/scripts/group_comparison_analysis.py \
  --smica COM_CMB_IQU-smica_2048_R3.00_full.fits \
  --nilc COM_CMB_IQU-nilc_2048_R3.00_full.fits \
  --commander COM_CMB_IQU-commander_2048_R3.00_full.fits \
  --sevem COM_CMB_IQU-sevem_2048_R3.00_full.fits \
  --outdir other_groups/results/ --nmc 10000

# Appendix D: Frequency-channel independence
python freq_channels/scripts/freq_channel_d3_analysis.py \
  --datadir freq_channels/data/ --outdir freq_channels/results/ --nmc 10000

# Appendix E: Mask-variation robustness
python mask_variation/scripts/mask_variation_analysis.py \
  --mapdir /path/to/planck/maps/ --maskdir mask_variation/masks/ \
  --galmaskfile HFI_Mask_GalPlane-apo0_2048_R2.00.fits \
  --outdir mask_variation/results/ --nmc 5000
```

## D3 Axis

All analyses use the fixed D3 symmetry axis at Galactic coordinates:

**(l, b) = (50.3 deg, -64.9 deg)**

This axis was determined by maximizing the A2 irrep fraction at l = 3 over an NSIDE = 32 HEALPix grid (12,288 directions), then refined. It is held fixed across all tests.

## Script-to-Paper Mapping

| Script | Paper Section | Output |
|--------|--------------|--------|
| `d3_extended_analysis.py` | Sections 3-5 | Tables 1-5, 9-10; Figures 1-6, 9 |
| `d3_parity_test.py` | Section 5.1 | Table 6 |
| `d3_witness_transfer_analysis.py` | Section 5.2 | Table 7; Figure 10 |
| `d3_factorization_robustness.py` | Section 5.3 | Table 8 |
| `d3_polarization_analysis.py` | Section 6.2 | EE null test |
| `d3_phase_locking_test.py` | Section 6.3 | Phase-locking null test |
| `wmap_d3_analysis.py` | Appendix B | WMAP cross-experiment validation |
| `group_comparison_analysis.py` | Appendix C | D3 vs D4/D5/D6 specificity |
| `freq_channel_d3_analysis.py` | Appendix D | Frequency independence |
| `mask_variation_analysis.py` | Appendix E | Mask robustness |

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@article{mereau2026d3cmb,
  title        = {Evidence for Dihedral $D_3$ Symmetry in the Planck CMB Temperature Anisotropy},
  author       = {Mereau, Robert},
  year         = {2026},
  month        = apr,
  publisher    = {Preprints},
  doi          = {10.20944/preprints202604.0264.v2},
  url          = {https://doi.org/10.20944/preprints202604.0264.v2},
  note         = {Preprint, version 2}
}
```
