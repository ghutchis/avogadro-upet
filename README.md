# avogadro-upet

Energy and gradient models for [Avogadro 2](https://two.avogadro.cc/) using
[UPET](https://github.com/lab-cosmo/upet), the Point-Edge Transformer (PET)
universal machine-learning interatomic potentials from the COSMO lab (EPFL).

Forces are always the gradient of the predicted energy (conservative). The
non-conservative direct-force heads that some PET checkpoints carry are never
used. Energies are reported in kJ/mol and gradients in kJ/mol/Angstrom.
Calculations run in float64 on CUDA if available, otherwise on the CPU
(never MPS).

## Models

Checkpoint versions are pinned in `src/avogadro_upet/_models.py`.

| Identifier | Checkpoint | Training data / level | Elements | Periodic |
|---|---|---|---|---|
| PET-MAD-S | `pet-mad-s` v1.6.0 | MAD, r2SCAN | 1-102 | yes |
| PET-SPICE | `pet-spice-s` v0.2.0 | SPICE, wB97M-D3 | H, C-F, P-Cl, Br, I | no |
| PET-OMat | `pet-omat-s` v1.0.0 | OMat24, PBE | 1-83, 89-94 | yes |

Element lists come from the checkpoints themselves: the atomic types of
PET-MAD-S and PET-SPICE, and for PET-OMat the elements with a non-zero
composition baseline (the checkpoint declares 102 atomic types but was trained
on 89 elements). The upstream docs quote 17 elements for PET-SPICE, but the
v0.2.0 checkpoint declares only the 10 listed here; the plugin follows the
checkpoint. PET-SPICE was trained on isolated molecules only, so periodic
systems are not offered.

None of these models take a total charge or spin multiplicity as input, so all
three are neutral, closed-shell only (`support.ions` and `support.radicals`
are false, so Avogadro will not offer them for charged or open-shell systems).

## Installation

Install through Avogadro's plugin manager. Weights download from Hugging Face
(`lab-cosmo/upet`) into the Hugging Face cache. Run the setup task once so the
first calculation does not stall:

    pixi run setup        # or: avogadro-upet-setup

Platforms: osx-arm64 and linux-64 are locked. Only osx-arm64 (CPU) has been
tested; linux-64 and CUDA are untested. Windows is not configured.

## Licenses

- This plugin: BSD-3-Clause.
- UPET code: BSD-3-Clause.
- Model weights (Hugging Face `lab-cosmo/upet`): the repository card declares
  `license: bsd-3-clause`. The training datasets (MAD, SPICE, OMat24) carry
  their own terms.

## Citation

- A. Mazitov et al., "PET-MAD as a lightweight universal interatomic
  potential for advanced materials modeling", Nature Communications (2025),
  https://doi.org/10.1038/s41467-025-65662-7
- UPET, https://arxiv.org/abs/2601.16195
- PET architecture, https://arxiv.org/abs/2305.19302
