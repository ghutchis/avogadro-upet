"""Shared fixtures for the avogadro-upet tests.

The helpers here speak the same binary protocol Avogadro does, against the real
entry point in a subprocess. That is deliberate: running the model in-process
would hide the most common way an energy plugin breaks, which is a dependency
printing to stdout on import and corrupting the frame stream.
"""

import io
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from avogadro.energy import (
    FLAG_BATCH_MODE,
    FLAG_REQUEST_ENERGY_AND_GRADIENT,
    FLAG_REQUEST_GRADIENT,
    FLAG_REQUEST_HESSIAN,
    FLAG_RESPONSE_ERROR,
    read_frame,
    write_frame,
)

DATA = Path(__file__).parent / "data"

IDENTIFIER = "PET-MAD-S"

# Run the installed entry point the way Avogadro does, but without depending on
# PATH: `python -c ...` leaves sys.argv[1:] as the arguments after the code.
_LAUNCH = [
    sys.executable,
    "-c",
    "from avogadro_upet import main; main()",
]


@pytest.fixture
def water():
    """The bootstrap JSON Avogadro would send for a water molecule."""
    with (DATA / "water.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def image_mask(bootstrap):
    """True for the atoms that are periodic images of another atom."""
    fractional = np.array(
        bootstrap["cjson"]["atoms"]["coords"]["3dFractional"], dtype=float
    ).reshape(-1, 3)
    return np.any(np.abs(fractional - 1.0) < 1e-4, axis=1)


def coordinates(bootstrap):
    """The molecule's coordinates as an (N, 3) array."""
    flat = bootstrap["cjson"]["atoms"]["coords"]["3d"]
    return np.array(flat, dtype=float).reshape(-1, 3)


def request_frame(coords, flags=0):
    """Encode one coordinate request the way Avogadro's C++ side does."""
    coords = np.ascontiguousarray(coords, dtype="<f8")
    buffer = io.BytesIO()
    if flags & FLAG_BATCH_MODE:
        atom_count = coords.shape[1]
        payload = struct.pack("<I", coords.shape[0]) + coords.tobytes()
    else:
        atom_count = coords.shape[0]
        payload = coords.tobytes()
    write_frame(buffer, flags, atom_count, payload)
    return buffer.getvalue()


def exchange(bootstrap, requests, identifier=IDENTIFIER):
    """Bootstrap the model, send every request, and decode the replies.

    `requests` is a list of (coords, flags) pairs. Returns a list of decoded
    responses; a response is a float for an energy, an (N, 3) array for a
    gradient, and a tuple for a fused energy and gradient.
    """
    atom_count = len(bootstrap["cjson"]["atoms"]["elements"]["number"])

    stdin = json.dumps(bootstrap).encode("utf-8") + b"\n"
    for coords, flags in requests:
        stdin += request_frame(coords, flags)

    process = subprocess.run(
        _LAUNCH + [identifier, "--protocol", "binary-v1"],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8", "replace")

    assert process.stdout[:4] == b"AVB1", process.stdout[:80]
    stream = io.BytesIO(process.stdout)
    responses = [
        _decode(read_frame(stream), atom_count, flags)
        for _, flags in requests
    ]

    # One response frame per request and nothing else. Anything left over is
    # something that wrote to stdout when it should have written to stderr --
    # a debugging print, or a dependency announcing itself on import. Python
    # block-buffers text stdout into a pipe, so such output often surfaces at
    # exit rather than in the middle of the frames, which is why it can look
    # harmless right up until a flush lands it between two frames.
    trailing = stream.read()
    assert not trailing, (
        "unexpected bytes on stdout after the last response frame: "
        + repr(trailing[:200])
    )

    return responses


def _decode(frame, atom_count, request_flags):
    assert frame is not None, "the model closed the stream early"

    if frame.flags & FLAG_RESPONSE_ERROR:
        raise AssertionError(
            "model reported: " + frame.payload.decode("utf-8", "replace")
        )

    if request_flags & FLAG_REQUEST_HESSIAN:
        dim = 3 * atom_count
        return np.frombuffer(frame.payload, dtype="<f8").reshape(dim, dim)

    if request_flags & FLAG_BATCH_MODE:
        batch_size = struct.unpack("<I", frame.payload[:4])[0]
        values = np.frombuffer(frame.payload[4:], dtype="<f8")
        if request_flags & FLAG_REQUEST_GRADIENT:
            return values.reshape(batch_size, atom_count, 3)
        return values

    if request_flags & FLAG_REQUEST_ENERGY_AND_GRADIENT:
        energy = struct.unpack("<d", frame.payload[:8])[0]
        gradient = np.frombuffer(frame.payload[8:], dtype="<f8")
        return energy, gradient.reshape(atom_count, 3)

    if request_flags & FLAG_REQUEST_GRADIENT:
        return np.frombuffer(frame.payload, dtype="<f8").reshape(atom_count, 3)

    return struct.unpack("<d", frame.payload)[0]
