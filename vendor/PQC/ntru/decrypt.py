"""NTRU + AES-256-GCM hybrid decryption."""

import numpy as np
import hashlib
from typing import Dict, List, Union, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
from .polynomial_ring import PolynomialRing


class NTRUDecryptor:
    """Decrypts embeddings and metadata encrypted by NTRUEncryptor."""

    def __init__(self, private_key: Dict):
        """Initialise with an NTRU private key {'f', 'Fp', 'N', 'p', 'q'}."""
        self.f = private_key['f']
        self.Fp = private_key['Fp']
        self.N = private_key['N']
        self.p = private_key['p']
        self.q = private_key['q']

        self.ring = PolynomialRing(self.N, self.q, self.p)

    def _ntru_decapsulate_key(self, ntru_ciphertext: np.ndarray,
                               seed_coeffs: np.ndarray = None) -> bytes:
        """Recover the 32-byte AES key from the NTRU ciphertext (or stored seed)."""
        if seed_coeffs is not None:
            seed_bytes = seed_coeffs.tobytes()
            return hashlib.sha256(seed_bytes).digest()

        a = self.ring.multiply(self.f, ntru_ciphertext, self.q)

        a = self.ring.mod_reduce(a, self.p)

        decrypted_seed = self.ring.multiply(self.Fp, a, self.p)

        seed_bytes = decrypted_seed.tobytes()
        return hashlib.sha256(seed_bytes).digest()

    def decrypt_embedding(self, encrypted_data: Dict) -> np.ndarray:
        """Decrypt an embedding, dispatching on its version."""
        version = encrypted_data.get('version', 1)

        if version >= 3:
            return self._decrypt_embedding_pqc(encrypted_data)
        elif 'encrypted_bytes' in encrypted_data:
            return self._decrypt_embedding_hybrid_v2(encrypted_data)
        else:
            return self._decrypt_embedding_legacy(encrypted_data)

    def _decrypt_embedding_pqc(self, encrypted_data: Dict) -> np.ndarray:
        """Decrypt a version-3 embedding (NTRU KEM + AES-256-GCM)."""
        ntru_ciphertext = encrypted_data['ntru_ciphertext']
        seed_coeffs = encrypted_data.get('seed_coeffs')
        aes_nonce = encrypted_data['aes_nonce']
        aes_ciphertext = encrypted_data['aes_ciphertext']
        aad = encrypted_data['aad']
        original_shape = encrypted_data['original_shape']
        original_norm = encrypted_data['original_norm']
        scale_factor = encrypted_data['scale_factor']

        aes_key = self._ntru_decapsulate_key(ntru_ciphertext, seed_coeffs)

        aesgcm = AESGCM(aes_key)
        try:
            embedding_bytes = aesgcm.decrypt(aes_nonce, aes_ciphertext, aad)
        except InvalidTag:
            raise ValueError("Decryption failed: Data integrity check failed (tampering detected)")

        quantized = np.frombuffer(embedding_bytes, dtype=np.int16)
        embedding = quantized.astype(np.float32) / scale_factor

        current_norm = np.linalg.norm(embedding)
        if current_norm > 0:
            embedding = embedding * (original_norm / current_norm)

        return embedding.reshape(original_shape)

    def _decrypt_embedding_hybrid_v2(self, encrypted_data: Dict) -> np.ndarray:
        """Decrypt a legacy version-2 embedding."""
        encrypted_bytes = encrypted_data['encrypted_bytes']
        original_shape = encrypted_data['original_shape']
        original_norm = encrypted_data['original_norm']
        scale_factor = encrypted_data['scale_factor']

        prng_seed = encrypted_data.get('prng_seed')
        if prng_seed is None:
            raise ValueError("No seed found in v2 encrypted data")

        np.random.seed(int(prng_seed))
        keystream = np.random.randint(0, 256, size=len(encrypted_bytes), dtype=np.uint8)

        encrypted_array = np.frombuffer(encrypted_bytes, dtype=np.uint8)
        decrypted_bytes = np.bitwise_xor(encrypted_array, keystream)

        quantized = np.frombuffer(decrypted_bytes.tobytes(), dtype=np.int16)
        embedding = quantized.astype(np.float32) / scale_factor

        current_norm = np.linalg.norm(embedding)
        if current_norm > 0:
            embedding = embedding * (original_norm / current_norm)

        return embedding.reshape(original_shape)

    def _decrypt_embedding_legacy(self, encrypted_data: Dict) -> np.ndarray:
        """Decrypt a legacy version-1 embedding."""
        ciphertexts = encrypted_data['ciphertexts']
        original_shape = encrypted_data['original_shape']
        scale_factor = encrypted_data['scale_factor']

        decrypted_chunks = []
        for ciphertext in ciphertexts:
            message = self.decrypt_polynomial(ciphertext)
            decrypted_chunks.append(message)

        quantized = self._reassemble_chunks(decrypted_chunks, original_shape)
        embedding = self._dequantize_embedding(quantized, scale_factor)

        return embedding

    def decrypt_polynomial(self, ciphertext: np.ndarray) -> np.ndarray:
        """Decrypt a single ciphertext polynomial (legacy)."""
        a = self.ring.multiply(self.f, ciphertext, self.q)
        a = self.ring.mod_reduce(a, self.p)
        message = self.ring.multiply(self.Fp, a, self.p)
        return message

    def decrypt_metadata(self, encrypted_meta: Dict) -> str:
        """Decrypt metadata, dispatching on its version."""
        version = encrypted_meta.get('version', 1)

        if version >= 4:
            return self._decrypt_metadata_pure_ntru(encrypted_meta)
        elif version >= 3:
            return self._decrypt_metadata_pqc(encrypted_meta)
        elif 'encrypted_bytes' in encrypted_meta:
            return self._decrypt_metadata_hybrid_v2(encrypted_meta)
        else:
            return self._decrypt_metadata_legacy(encrypted_meta)

    def _decrypt_metadata_pure_ntru(self, encrypted_meta: Dict) -> str:
        """Decrypt version-4 metadata (pure NTRU, no AES)."""
        from .pure_ntru import PureNTRUDecryptor

        pure_decryptor = PureNTRUDecryptor({
            'f': self.f,
            'Fp': self.Fp,
            'N': self.N,
            'p': self.p,
            'q': self.q
        })

        decrypted_bytes = pure_decryptor.decrypt_bytes(encrypted_meta)
        return decrypted_bytes.decode('utf-8')

    def _decrypt_metadata_pqc(self, encrypted_meta: Dict) -> str:
        """Decrypt version-3 metadata (NTRU KEM + AES-256-GCM)."""
        ntru_ciphertext = encrypted_meta['ntru_ciphertext']
        seed_coeffs = encrypted_meta.get('seed_coeffs')
        aes_nonce = encrypted_meta['aes_nonce']
        aes_ciphertext = encrypted_meta['aes_ciphertext']
        aad = encrypted_meta['aad']

        aes_key = self._ntru_decapsulate_key(ntru_ciphertext, seed_coeffs)

        aesgcm = AESGCM(aes_key)
        try:
            metadata_bytes = aesgcm.decrypt(aes_nonce, aes_ciphertext, aad)
        except InvalidTag:
            raise ValueError("Decryption failed: Metadata integrity check failed")

        return metadata_bytes.decode('utf-8')

    def _decrypt_metadata_hybrid_v2(self, encrypted_meta: Dict) -> str:
        """Decrypt legacy version-2 metadata."""
        encrypted_bytes = encrypted_meta['encrypted_bytes']

        prng_seed = encrypted_meta.get('prng_seed')
        if prng_seed is None:
            raise ValueError("No seed found in v2 encrypted metadata")

        np.random.seed(int(prng_seed))
        keystream = np.random.randint(0, 256, size=len(encrypted_bytes), dtype=np.uint8)

        encrypted_array = np.frombuffer(encrypted_bytes, dtype=np.uint8)
        decrypted_bytes = np.bitwise_xor(encrypted_array, keystream)

        try:
            return decrypted_bytes.tobytes().decode('utf-8').rstrip('\x00')
        except UnicodeDecodeError:
            return decrypted_bytes.tobytes().decode('utf-8', errors='ignore').rstrip('\x00')

    def _decrypt_metadata_legacy(self, encrypted_meta: Dict) -> str:
        """Decrypt legacy version-1 metadata."""
        ciphertexts = encrypted_meta['ciphertexts']
        original_length = encrypted_meta['original_length']
        chunk_size = self.N - 1

        decrypted_bytes = []
        for ciphertext in ciphertexts:
            message = self.decrypt_polynomial(ciphertext)
            for coeff in message[:chunk_size]:
                byte_val = int(coeff) % 256
                decrypted_bytes.append(byte_val)

        decrypted_bytes = bytes(decrypted_bytes[:original_length])

        try:
            return decrypted_bytes.decode('utf-8').rstrip('\x00')
        except UnicodeDecodeError:
            return decrypted_bytes.decode('utf-8', errors='ignore').rstrip('\x00')

    def _reassemble_chunks(self, chunks: List[np.ndarray], original_shape: tuple) -> np.ndarray:
        """Reassemble decrypted chunks into the original shape (legacy)."""
        total_len = np.prod(original_shape)
        chunk_size = self.N - 1

        result = []
        for chunk in chunks:
            result.extend(chunk[:chunk_size].tolist())

        result = np.array(result[:total_len], dtype=np.int64)
        return result.reshape(original_shape)

    def _dequantize_embedding(self, quantized: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale integer coefficients back to floats (legacy)."""
        return quantized.astype(np.float32) / scale_factor

    def decrypt_all_embeddings(self, encrypted_embeddings: List[Dict]) -> List[np.ndarray]:
        """Decrypt a list of encrypted embeddings."""
        return [self.decrypt_embedding(enc) for enc in encrypted_embeddings]


def decrypt_embedding(encrypted_data: Dict, private_key: Dict) -> np.ndarray:
    """Decrypt an embedding with the given private key."""
    decryptor = NTRUDecryptor(private_key)
    return decryptor.decrypt_embedding(encrypted_data)


if __name__ == "__main__":
    from keygen import NTRUKeyGenerator
    from encrypt import NTRUEncryptor

    print("Testing NTRU + AES-256-GCM Hybrid Encryption/Decryption...")
    print("=" * 60)

    generator = NTRUKeyGenerator()
    public_key, private_key = generator.generate_keys()

    original_embedding = np.random.randn(512).astype(np.float32)
    original_embedding = original_embedding / np.linalg.norm(original_embedding)

    print(f"Original embedding shape: {original_embedding.shape}")
    print(f"Original first 5 values: {original_embedding[:5]}")

    encryptor = NTRUEncryptor(public_key)
    encrypted = encryptor.encrypt_embedding(original_embedding)
    print(f"\nEncryption successful (Version {encrypted['version']})")

    decryptor = NTRUDecryptor(private_key)
    decrypted_embedding = decryptor.decrypt_embedding(encrypted)

    print(f"\nDecrypted embedding shape: {decrypted_embedding.shape}")
    print(f"Decrypted first 5 values: {decrypted_embedding[:5]}")

    cosine_sim = np.dot(original_embedding, decrypted_embedding) / (
        np.linalg.norm(original_embedding) * np.linalg.norm(decrypted_embedding)
    )
    print(f"\nCosine similarity (original vs decrypted): {cosine_sim:.6f}")

    if cosine_sim > 0.999:
        print("SUCCESS: True PQC encryption/decryption works!")
    else:
        print("WARNING: Similarity is low")

    print("\n" + "=" * 60)
    print("Testing Metadata Encryption/Decryption...")

    encrypted_meta = encryptor.encrypt_metadata("John Doe", "user_001")
    decrypted_meta = decryptor.decrypt_metadata(encrypted_meta)
    print(f"Original: 'John Doe|user_001'")
    print(f"Decrypted: '{decrypted_meta}'")

    if decrypted_meta == "John Doe|user_001":
        print("SUCCESS: Metadata encryption/decryption works!")
    else:
        print("WARNING: Metadata mismatch")
