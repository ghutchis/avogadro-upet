#  This source file is part of the Avogadro project
#  This source code is released under the BSD-3-Clause License, (see "LICENSE").
#  https://github.com/ghutchis/avogadro-upet/

"""Energy models for UPET (PET-MAD, PET-SPICE, PET-OMat), binary-v1 protocol.

Avogadro works in kJ/mol and Angstrom, and wants the *gradient* dE/dx, not the
force. Forces come from the energy gradient (conservative); the
non-conservative force head is never used.
"""

import json
import sys

import numpy as np
import torch
from ase import Atoms
from ase.cell import Cell

from ._models import MODELS

from avogadro.energy import EnergyServer

# ASE reports eV and eV/Angstrom.
_EV_TO_KJ_MOL = 96.4853321

# Don't use MPS on Apple: no float64.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_calculator(identifier):
    """Build the pinned UPET ASE calculator for an Avogadro identifier."""
    model, version = MODELS[identifier]

    from upet.ase import UPETCalculator

    # non_conservative=False: forces are the gradient of the energy.
    return UPETCalculator(
        model=model,
        version=version,
        device=_DEVICE,
        dtype=torch.float64,
        non_conservative=False,
        uncertainty_threshold=None,
    )


def _prepare(identifier, mol_cjson):
    numbers = np.array(mol_cjson["atoms"]["elements"]["number"])
    coords = np.array(
        mol_cjson["atoms"]["coords"]["3d"], dtype=float
    ).reshape(-1, 3)

    # num_atoms must be the full count Avogadro sent, images included.
    num_atoms = len(numbers)

    # Periodic image atoms have fractional coordinates ~1.0 and duplicate the
    # atoms at 0.0; drop them from the calculation and zero their gradients.
    keep = np.ones(num_atoms, dtype=bool)

    if "unitCell" in mol_cjson:
        fractional = mol_cjson["atoms"]["coords"].get("3dFractional", [])
        if fractional:
            frac = np.array(fractional, dtype=float).reshape(-1, 3)
            keep = ~np.any(np.abs(frac - 1.0) < 1e-4, axis=1)

        lattice = np.array(
            mol_cjson["unitCell"]["cellVectors"], dtype=float
        ).reshape(3, 3)
        atoms = Atoms(numbers[keep], coords[keep], cell=Cell(lattice), pbc=True)
    else:
        atoms = Atoms(numbers[keep], coords[keep], pbc=False)

    atoms.calc = make_calculator(identifier)
    return {"atoms": atoms, "keep": keep, "num_atoms": num_atoms}


def _energy(model, coords):
    model["atoms"].set_positions(coords[model["keep"]])
    return float(model["atoms"].get_potential_energy()) * _EV_TO_KJ_MOL


def _energy_and_gradient(model, coords):
    atoms = model["atoms"]
    atoms.set_positions(coords[model["keep"]])

    energy = float(atoms.get_potential_energy()) * _EV_TO_KJ_MOL

    # force = -gradient; image atoms get zeros.
    gradient = np.zeros((model["num_atoms"], 3), dtype=np.float64)
    gradient[model["keep"]] = atoms.get_forces() * -_EV_TO_KJ_MOL
    return energy, gradient


def _respond(model, request):
    if request.wants_energy_and_gradient:
        energy, gradient = _energy_and_gradient(model, request.coords)
        request.send_energy_and_gradient(energy, gradient)
    elif request.wants_gradient:
        request.send_gradient(_energy_and_gradient(model, request.coords)[1])
    else:
        request.send_energy(_energy(model, request.coords))


def run(identifier, proto_out):
    # Read the bootstrap line from the binary buffer: the text wrapper would
    # swallow the first binary frame.
    bootstrap = json.loads(sys.stdin.buffer.readline())

    # UPET models take no charge or spin input, so they are neutral-only
    # (support.ions / support.radicals = false); charge and spin are ignored.
    model = _prepare(identifier, bootstrap["cjson"])

    with EnergyServer(
        sys.stdin.buffer, proto_out, model["num_atoms"]
    ) as server:
        for request in server.requests():
            try:
                _respond(model, request)
            except Exception as error:  # noqa: BLE001
                request.send_error(str(error))
