"""Cognitron: from-scratch Particle Neural Network + Hyperdimensional Computing.

No pretrained models. No FAISS. No pgvector. Every operation built from primitives.
"""

from cognitron.hdc import HDCEncoder, bind, bundle, permute, hamming, cosine
from cognitron.pnn import Particle, ParticleNetwork
from cognitron.pgd import ParticleGradientDescent, PGDConfig, SGDFallback, SGDConfig
from cognitron.geometric_index import GeometricIndex

__all__ = [
    "HDCEncoder",
    "bind",
    "bundle",
    "permute",
    "hamming",
    "cosine",
    "Particle",
    "ParticleNetwork",
    "ParticleGradientDescent",
    "PGDConfig",
    "SGDFallback",
    "SGDConfig",
    "GeometricIndex",
]

__version__ = "0.1.0"
