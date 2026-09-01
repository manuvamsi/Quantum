"""
PQC (Post-Quantum Cryptography) — deployment subset.

This is a TRIMMED copy for the Testing_App_deployment package. It ships only the
pieces the edge app actually calls at runtime:

- NTRU key generation + encryption/decryption  (`.ntru`)
- Secure metadata storage: NTRU + AES-256-GCM for name/phone/age  (`.metadata_storage`)
- Timing instrumentation                        (`.timing`)

The high-level `secure_login` / `secure_recognition` / `secure_registration` /
`secure_storage` helpers from the full project are intentionally NOT included
(they pull in the V1 training/login chain that this package does not need).
The engine imports the submodules directly, e.g.

    from PQC.metadata_storage import get_metadata_storage
    from PQC.ntru import NTRUKeyGenerator
"""

from .ntru import NTRUKeyGenerator, NTRUEncryptor, NTRUDecryptor
from .metadata_storage import SecureMetadataStorage, get_metadata_storage
from .timing import TimingLogger, Timer, get_timing_logger

__all__ = [
    # NTRU core
    "NTRUKeyGenerator",
    "NTRUEncryptor",
    "NTRUDecryptor",
    # Metadata storage
    "SecureMetadataStorage",
    "get_metadata_storage",
    # Timing
    "TimingLogger",
    "Timer",
    "get_timing_logger",
]
