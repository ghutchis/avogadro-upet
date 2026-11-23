#  This source file is part of the Avogadro project
#  This source code is released under the BSD-3-Clause License, (see "LICENSE").
#  https://github.com/ghutchis/avogadro-upet/

"""Entry point for the avogadro-upet plugin.

Avogadro calls this as:
    avogadro-upet <identifier> --protocol binary-v1 [--lang <locale>] [--debug]

with the molecule bootstrap JSON on stdin as one compact line, followed by
binary coordinate frames for as long as the model stays loaded.
"""

import argparse
import os


def setup():
    """Download and cache every exposed UPET checkpoint.

    Weights come from the Hugging Face repository lab-cosmo/upet and land in
    the Hugging Face cache. Each model is run once on water so that the first
    energy evaluation in Avogadro is not the slow one.
    """
    from ase.build import molecule

    from ._models import MODELS
    from ._upet import make_calculator

    for identifier, (model, version) in MODELS.items():
        print(f"Preparing {identifier} ({model} v{version})...", flush=True)
        atoms = molecule("H2O")
        atoms.calc = make_calculator(identifier)
        atoms.get_potential_energy()
    print("UPET setup complete.")


def main():
    parser = argparse.ArgumentParser("avogadro-upet")
    # Avogadro always passes the feature identifier as the first argument.
    parser.add_argument("feature")
    # Every plugin must accept these without crashing, even if it ignores them.
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--debug", action="store_true")
    # Sent for any energy model declaring protocol = "binary-v1".
    parser.add_argument("--protocol", nargs="?", default="binary-v1")
    args = parser.parse_args()

    # Handlers are imported inside each branch so that a slow dependency is
    # only loaded by the feature that actually needs it.
    from ._models import MODELS

    if args.feature in MODELS:
        run_model(args.feature)


def run_model(identifier):
    """Serve one model, with the protocol stream protected at the fd level.

    ML libraries print to stdout (including from C/C++ code) while importing
    and while building calculators, and Avogadro never resynchronizes a
    corrupted stream. So keep a private duplicate of the real stdout for
    protocol frames and point file descriptor 1 at stderr before importing
    anything heavy.
    """
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)

    from ._upet import run

    run(identifier, proto_out)
