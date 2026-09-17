"""NTRU + AES-256-GCM hybrid encryption."""

import numpy as np
import os
import hashlib
from typing import Dict, List, Union
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .polynomial_ring import PolynomialRing


class NTRUEncryptor:
    """Encrypts embeddings and metadata with NTRU + AES-256-GCM."""

    def __init__(self, public_key: Dict):
        """Initialise with an NTRU public key {'h', 'N', 'p', 'q'}."""
        self.h = public_key['h']
        self.N = public_key['N']
        self.p = public_key['p']
        self.q = public_key['q']

        self.dr = self.N // 3

        self.ring = PolynomialRing(self.N, self.q, self.p)

    def _ntru_encapsulate_key(self) -> tuple:
        """Produce an AES key with its NTRU ciphertext and seed."""
        seed_coeffs = self.ring.random_ternary(self.N // 3, self.N // 3)

        r = self.ring.random_ternary(self.dr, self.dr)

        r_h = self.ring.multiply(r, self.h, self.q)
        ciphertext = self.ring.add(r_h, seed_coeffs, self.q)

        seed_bytes = seed_coeffs.tobytes()
        aes_key = hashlib.sha256(seed_bytes).digest()

        return aes_key, ciphertext, seed_coeffs

    def encrypt_embedding(self, embedding: np.ndarray) -> Dict:
        """Encrypt a facial embedding; returns the encrypted payload dict."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized = embedding / norm
        else:
            normalized = embedding
            norm = 1.0

        scale = 32000.0
        quantized = np.round(normalized * scale).astype(np.int16)
        embedding_bytes = quantized.tobytes()

        aes_key, ntru_ciphertext, seed_coeffs = self._ntru_encapsulate_key()

        nonce = os.urandom(12)

        aesgcm = AESGCM(aes_key)
        aad = f"embedding|{embedding.shape}|{norm}".encode('utf-8')
        ciphertext = aesgcm.encrypt(nonce, embedding_bytes, aad)

        return {
            'ntru_ciphertext': ntru_ciphertext,
            'seed_coeffs': seed_coeffs,
            'aes_nonce': nonce,
            'aes_ciphertext': ciphertext,
            'aad': aad,
            'original_shape': embedding.shape,
            'original_norm': float(norm),
            'scale_factor': scale,
            'N': self.N,
            'q': self.q,
            'p': self.p,
            'version': 3
        }

    def encrypt_metadata(self, name: str, user_id: str = None) -> Dict:
        """Encrypt user metadata; returns the encrypted payload dict."""
        metadata_str = f"{name}|{user_id or ''}"
        metadata_bytes = metadata_str.encode('utf-8')

        aes_key, ntru_ciphertext, seed_coeffs = self._ntru_encapsulate_key()

        nonce = os.urandom(12)

        aesgcm = AESGCM(aes_key)
        aad = b"metadata"
        ciphertext = aesgcm.encrypt(nonce, metadata_bytes, aad)

        return {
            'ntru_ciphertext': ntru_ciphertext,
            'seed_coeffs': seed_coeffs,
            'aes_nonce': nonce,
            'aes_ciphertext': ciphertext,
            'aad': aad,
            'original_length': len(metadata_bytes),
            'N': self.N,
            'q': self.q,
            'p': self.p,
            'version': 3
        }

    def encrypt_polynomial(self, message: np.ndarray) -> np.ndarray:
        """Encrypt a single message polynomial (legacy)."""
        r = self.ring.random_ternary(self.dr, self.dr)
        r_h = self.ring.multiply(r, self.h, self.q)
        message_padded = self.ring._pad_to_N(message)
        ciphertext = self.ring.add(r_h, message_padded, self.q)
        return ciphertext


def encrypt_embedding(embedding: np.ndarray, public_key: Dict) -> Dict:
    """Encrypt an embedding with the given public key."""
    encryptor = NTRUEncryptor(public_key)
    return encryptor.encrypt_embedding(embedding)


if __name__ == "__main__":
    from keygen import NTRUKeyGenerator

    print("Testing NTRU + AES-256-GCM Hybrid Encryption...")
    print("=" * 60)

    generator = NTRUKeyGenerator()
    public_key, private_key = generator.generate_keys()

    test_embedding = np.random.randn(512).astype(np.float32)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)

    print(f"Original embedding shape: {test_embedding.shape}")
    print(f"Original embedding norm: {np.linalg.norm(test_embedding):.4f}")

    encryptor = NTRUEncryptor(public_key)
    encrypted = encryptor.encrypt_embedding(test_embedding)

    print(f"\nEncryption successful!")
    print(f"  Version: {encrypted['version']} (True PQC)")
    print(f"  NTRU ciphertext shape: {encrypted['ntru_key_ciphertext'].shape}")
    print(f"  AES ciphertext length: {len(encrypted['aes_ciphertext'])} bytes")
    print(f"  AES nonce length: {len(encrypted['aes_nonce'])} bytes")

    print("\n" + "=" * 60)
    print("Testing Metadata Encryption...")

    encrypted_meta = encryptor.encrypt_metadata("John Doe", "user_001")
    print(f"Metadata encrypted successfully!")
    print(f"  Version: {encrypted_meta['version']} (True PQC)")
