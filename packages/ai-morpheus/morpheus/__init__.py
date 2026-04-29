"""Morpheus: 3D synesthetic Neural Cellular Automaton.

Extends Mordvintsev et al. 2D Growing NCA into 3D with three jointly-trained
output modalities: RGB color, geometry (alpha), and per-cell audio frequency.
The audio modality is the genuine novelty per the novelty-research agent's
findings — published 3D NCAs cover geometry and color but not learned sound.
"""

from morpheus.nca3d import (
    NCA3D,
    NCAConfig,
    seed_grid,
    extract_voxels,
    extract_audio_field,
)

__all__ = [
    "NCA3D",
    "NCAConfig",
    "seed_grid",
    "extract_voxels",
    "extract_audio_field",
]

__version__ = "0.1.0"
