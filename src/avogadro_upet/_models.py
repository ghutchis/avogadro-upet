#  This source file is part of the Avogadro project
#  This source code is released under the BSD-3-Clause License, (see "LICENSE").
#  https://github.com/ghutchis/avogadro-upet/

"""The UPET checkpoints this plugin exposes, pinned to exact versions.

Each entry names the upstream model ("pet-mad-s") and the exact checkpoint
version. Elements for each model are declared in pyproject.toml
(support.elements); the README records how they were derived.
"""

# identifier -> (upstream model name, pinned checkpoint version)
MODELS = {
    "PET-MAD-S": ("pet-mad-s", "1.6.0"),
    "PET-SPICE": ("pet-spice-s", "0.2.0"),
    "PET-OMat": ("pet-omat-s", "1.0.0"),
}
