"""
NTRU Post-Quantum Cryptography Implementation

NTRU is a lattice-based public-key cryptosystem that is resistant to
attacks by quantum computers.

Ring: R = Z[X]/(X^N - 1)
Parameters: N=509, p=3, q=2048 (standard secure parameters)

Encryption Modes:
- Hybrid (v3): NTRU KEM + AES-256-GCM (NTRUEncryptor, NTRUDecryptor)
- Pure NTRU (v4): Direct NTRU polynomial encryption (PureNTRUEncryptor, PureNTRUDecryptor)

References:
- NTRU: A Ring-Based Public Key Cryptosystem (Hoffstein, Pipher, Silverman)
- NIST Post-Quantum Cryptography Standardization
"""

from .keygen import NTRUKeyGenerator
from .encrypt import NTRUEncryptor
from .decrypt import NTRUDecryptor
from .polynomial_ring import PolynomialRing
from .pure_ntru import PureNTRUEncryptor, PureNTRUDecryptor

__all__ = [
    'NTRUKeyGenerator',
    'NTRUEncryptor',
    'NTRUDecryptor',
    'PolynomialRing',
    'PureNTRUEncryptor',
    'PureNTRUDecryptor'
]
