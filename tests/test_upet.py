"""Tests for the UPET energy models in avogadro-upet.

Every case drives the real entry point over the binary protocol. Reference
values come from a direct UPETCalculator call with the same pinned checkpoint.
"""

import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import PropertyNotImplementedError  # noqa: F401

from avogadro.energy import (
    FLAG_REQUEST_ENERGY_AND_GRADIENT,
    FLAG_REQUEST_GRADIENT,
)

from avogadro_upet._models import MODELS
from conftest import coordinates, exchange, image_mask

EV = 96.4853321

MOLECULAR = ["PET-MAD-S", "PET-SPICE"]
PERIODIC = ["PET-MAD-S", "PET-OMat"]


def _reference(identifier, atoms):
    from avogadro_upet._upet import make_calculator

    atoms = atoms.copy()
    atoms.calc = make_calculator(identifier)
    return (
        atoms.get_potential_energy() * EV,
        -atoms.get_forces() * EV,
    )


def _with_images(atoms):
    """An Avogadro-style bootstrap: append images of atoms on the cell origin
    planes at fractional coordinate 1.0, as Avogadro does."""
    frac = atoms.get_scaled_positions(wrap=False)
    cell = np.array(atoms.cell)
    numbers = list(atoms.numbers)
    fractional = [f for f in frac]
    for i, f in enumerate(frac):
        shifts = [
            np.array(s, dtype=float)
            for s in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
            if all(abs(f[k]) < 1e-8 or s[k] == 0 for k in range(3))
            and any(s[k] and abs(f[k]) < 1e-8 for k in range(3))
        ]
        for s in shifts:
            numbers.append(numbers[i])
            fractional.append(f + s)
    fractional = np.array(fractional)
    cart = fractional @ cell
    return {
        "cjson": {
            "chemicalJson": 1,
            "atoms": {
                "elements": {"number": [int(n) for n in numbers]},
                "coords": {
                    "3d": cart.ravel().tolist(),
                    "3dFractional": fractional.ravel().tolist(),
                },
            },
            "unitCell": {"cellVectors": cell.ravel().tolist()},
        },
        "charge": 0,
        "spin": 1,
        "options": {},
        "selectedAtoms": [],
    }, len(atoms)


def _nacl():
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    rng = np.random.default_rng(1)
    noise = rng.normal(scale=0.05, size=atoms.positions.shape)
    # Keep atom 0 exactly on the origin so Avogadro-style images exist.
    noise[0] = 0.0
    atoms.positions += noise
    return atoms


def _silicon():
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    rng = np.random.default_rng(2)
    noise = rng.normal(scale=0.05, size=atoms.positions.shape)
    # Keep atom 0 exactly on the origin so Avogadro-style images exist.
    noise[0] = 0.0
    atoms.positions += noise
    return atoms


def test_identifiers_match_pyproject():
    root = Path(__file__).parent.parent
    meta = tomllib.loads((root / "pyproject.toml").read_text())
    declared = {m["identifier"] for m in meta["tool"]["avogadro"]["energy-models"]}
    assert declared == set(MODELS)
    for model in meta["tool"]["avogadro"]["energy-models"]:
        assert model["support"]["ions"] is False
        assert model["support"]["radicals"] is False


@pytest.mark.parametrize("identifier", MOLECULAR)
def test_water_matches_reference(water, identifier):
    coords = coordinates(water)
    energy, gradient, fused = exchange(
        water,
        [
            (coords, 0),
            (coords, FLAG_REQUEST_GRADIENT),
            (coords, FLAG_REQUEST_ENERGY_AND_GRADIENT),
        ],
        identifier=identifier,
    )
    ref_e, ref_g = _reference(identifier, Atoms("OH2", positions=coords))

    assert energy == pytest.approx(ref_e, rel=1e-6)
    assert fused[0] == pytest.approx(ref_e, rel=1e-6)
    assert gradient == pytest.approx(ref_g, rel=1e-6, abs=1e-6)
    assert fused[1] == pytest.approx(ref_g, rel=1e-6, abs=1e-6)
    # No net force on an isolated molecule.
    assert np.abs(gradient.sum(axis=0)).max() < 1e-4


@pytest.mark.parametrize("identifier", MOLECULAR)
def test_water_gradient_matches_finite_differences(water, identifier):
    coords = coordinates(water)
    step = 1.0e-3
    requests = [(coords, FLAG_REQUEST_GRADIENT)]
    for atom in range(coords.shape[0]):
        for axis in range(3):
            for sign in (+1, -1):
                displaced = coords.copy()
                displaced[atom, axis] += sign * step
                requests.append((displaced, 0))
    results = exchange(water, requests, identifier=identifier)
    analytic, energies = results[0], results[1:]
    numerical = np.empty_like(analytic)
    for index in range(analytic.size):
        numerical.flat[index] = (
            energies[2 * index] - energies[2 * index + 1]
        ) / (2.0 * step)
    err = np.abs(analytic - numerical).max()
    print(identifier, "FD max error", err)
    assert analytic == pytest.approx(numerical, rel=1e-3, abs=5e-3)


@pytest.mark.parametrize("identifier", PERIODIC)
@pytest.mark.parametrize("builder", [_nacl, _silicon])
def test_crystal_matches_reference_with_images(identifier, builder):
    atoms = builder()
    bootstrap, n_real = _with_images(atoms)
    coords = coordinates(bootstrap)
    assert len(coords) > n_real  # image atoms are present

    energy, fused = exchange(
        bootstrap,
        [(coords, 0), (coords, FLAG_REQUEST_ENERGY_AND_GRADIENT)],
        identifier=identifier,
    )
    ref_e, ref_g = _reference(identifier, atoms)
    images = image_mask(bootstrap)

    assert energy == pytest.approx(ref_e, rel=1e-6)
    assert fused[0] == pytest.approx(ref_e, rel=1e-6)
    assert fused[1].shape == coords.shape
    assert np.all(fused[1][images] == 0.0)
    assert fused[1][~images] == pytest.approx(ref_g, rel=1e-6, abs=1e-6)


def test_gradient_covers_every_atom_sent():
    bootstrap, _ = _with_images(_nacl())
    coords = coordinates(bootstrap)
    (gradient,) = exchange(
        bootstrap, [(coords, FLAG_REQUEST_GRADIENT)], identifier="PET-OMat"
    )
    assert gradient.shape == coords.shape
    assert np.all(np.isfinite(gradient))


def test_model_stays_up_and_is_translation_invariant(water):
    coords = coordinates(water)
    stretched = coords.copy()
    stretched[1] *= 1.1
    first, second, again, moved = exchange(
        water,
        [(coords, 0), (stretched, 0), (coords, 0), (coords + 3.0, 0)],
        identifier="PET-SPICE",
    )
    assert first != pytest.approx(second)
    assert first == pytest.approx(again)
    assert first == pytest.approx(moved, rel=1e-8)
