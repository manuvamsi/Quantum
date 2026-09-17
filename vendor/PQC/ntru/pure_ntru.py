"""Pure NTRU encryption/decryption (no AES layer)."""

import numpy as np
from typing import Dict, Optional, List
from .polynomial_ring import PolynomialRing


class PureNTRUEncryptor:
    """Encrypts arbitrary bytes/metadata directly with NTRU, chunked as needed."""

    def __init__(self, public_key: Dict):
        """Initialise with an NTRU public key {'h', 'N', 'p', 'q'}."""
        self.h = public_key['h']
        self.N = public_key['N']
        self.p = public_key['p']
        self.q = public_key['q']

        self.dr = self.N // 3

        self.ring = PolynomialRing(self.N, self.q, self.p)

        self.bytes_per_poly = (self.N - 18) // 6

    def _bytes_to_ternary(self, data: bytes) -> np.ndarray:
        """Encode bytes as ternary coefficients."""
        coeffs = []
        for byte in data:
            val = byte
            for _ in range(6):
                digit = val % 3
                coeffs.append(digit - 1)
                val //= 3
        return np.array(coeffs, dtype=np.int64)

    def _ternary_to_bytes(self, coeffs: np.ndarray) -> bytes:
        """Decode ternary coefficients back to bytes."""
        result = []
        for i in range(0, len(coeffs), 6):
            chunk = coeffs[i:i+6]
            if len(chunk) < 6:
                break
            val = 0
            for j in range(5, -1, -1):
                coeff_val = int(chunk[j])
                if coeff_val < -1:
                    coeff_val = -1
                elif coeff_val > 1:
                    coeff_val = 1
                val = val * 3 + (coeff_val + 1)
            result.append(val % 256)
        return bytes(result)

    def encrypt_polynomial(self, message_poly: np.ndarray) -> np.ndarray:
        """Encrypt a single message polynomial."""
        r = self.ring.random_ternary(self.dr, self.dr)

        r_h = self.ring.multiply(r, self.h, self.q)

        message_padded = self.ring._pad_to_N(message_poly)

        ciphertext = self.ring.add(r_h, message_padded, self.q)

        return ciphertext

    def encrypt_bytes(self, data: bytes) -> Dict:
        """Encrypt arbitrary bytes; returns the encrypted payload dict."""
        checksum = self._compute_checksum(data)

        length = len(data)
        header = bytes([
            length & 0xFF,
            (length >> 8) & 0xFF,
            checksum
        ])

        full_data = header + data

        ternary = self._bytes_to_ternary(full_data)

        ciphertexts = []
        for i in range(0, len(ternary), self.N):
            chunk = ternary[i:i + self.N]
            if len(chunk) < self.N:
                chunk = np.pad(chunk, (0, self.N - len(chunk)), mode='constant')
            ciphertext = self.encrypt_polynomial(chunk)
            ciphertexts.append(ciphertext)

        return {
            'ciphertexts': ciphertexts,
            'num_chunks': len(ciphertexts),
            'original_length': length,
            'N': self.N,
            'p': self.p,
            'q': self.q,
            'version': 4
        }

    def encrypt_metadata(self, metadata_dict: Dict) -> Dict:
        """Encrypt a metadata dictionary (serialised to JSON)."""
        import json

        metadata_json = json.dumps(metadata_dict, ensure_ascii=False)
        metadata_bytes = metadata_json.encode('utf-8')

        return self.encrypt_bytes(metadata_bytes)

    def _compute_checksum(self, data: bytes) -> int:
        """Return a one-byte integrity checksum for the data."""
        checksum = 0
        for i, byte in enumerate(data):
            checksum ^= ((byte << (i % 8)) | (byte >> (8 - i % 8))) & 0xFF
        return checksum


class PureNTRUDecryptor:
    """Decrypts data produced by PureNTRUEncryptor."""

    def __init__(self, private_key: Dict):
        """Initialise with an NTRU private key {'f', 'Fp', 'N', 'p', 'q'}."""
        self.f = private_key['f']
        self.Fp = private_key['Fp']
        self.N = private_key['N']
        self.p = private_key['p']
        self.q = private_key['q']

        self.ring = PolynomialRing(self.N, self.q, self.p)

    def decrypt_polynomial(self, ciphertext: np.ndarray) -> np.ndarray:
        """Decrypt a single ciphertext polynomial."""
        a = self.ring.multiply(self.f, ciphertext, self.q)

        a = self.ring.mod_reduce(a, self.p)

        message = self.ring.multiply(self.Fp, a, self.p)

        return message

    def decrypt_bytes(self, encrypted: Dict) -> Optional[bytes]:
        """Decrypt an encrypt_bytes() payload and verify its checksum."""
        ciphertexts = encrypted['ciphertexts']
        original_length = encrypted.get('original_length', 0)

        all_coeffs = []
        for ciphertext in ciphertexts:
            message = self.decrypt_polynomial(ciphertext)
            all_coeffs.extend(message.tolist())

        coeffs = np.array(all_coeffs, dtype=np.int64)
        decrypted = self._ternary_to_bytes(coeffs)

        if len(decrypted) < 3:
            raise ValueError("Decrypted data too short - missing header")

        length = decrypted[0] | (decrypted[1] << 8)
        stored_checksum = decrypted[2]
        data = decrypted[3:3 + length]

        if len(data) != length:
            raise ValueError(f"Length mismatch: expected {length}, got {len(data)}")

        computed_checksum = self._compute_checksum(data)
        if computed_checksum != stored_checksum:
            raise ValueError(
                f"Data integrity check failed: "
                f"expected checksum {stored_checksum}, got {computed_checksum}"
            )

        return bytes(data)

    def decrypt_metadata(self, encrypted: Dict) -> Optional[Dict]:
        """Decrypt an encrypt_metadata() payload back to a dict."""
        import json

        decrypted_bytes = self.decrypt_bytes(encrypted)
        if decrypted_bytes is None:
            return None

        try:
            metadata_json = decrypted_bytes.decode('utf-8')
            return json.loads(metadata_json)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to parse decrypted metadata: {e}")

    def _ternary_to_bytes(self, coeffs: np.ndarray) -> bytes:
        """Decode ternary coefficients back to bytes."""
        result = []
        for i in range(0, len(coeffs), 6):
            chunk = coeffs[i:i+6]
            if len(chunk) < 6:
                break
            val = 0
            for j in range(5, -1, -1):
                coeff_val = int(chunk[j])
                if coeff_val < -1:
                    coeff_val = -1
                elif coeff_val > 1:
                    coeff_val = 1
                val = val * 3 + (coeff_val + 1)
            result.append(val % 256)
        return bytes(result)

    def _compute_checksum(self, data: bytes) -> int:
        """Return a one-byte integrity checksum matching the encryptor."""
        checksum = 0
        for i, byte in enumerate(data):
            checksum ^= ((byte << (i % 8)) | (byte >> (8 - i % 8))) & 0xFF
        return checksum


def encrypt_bytes_pure_ntru(data: bytes, public_key: Dict) -> Dict:
    """Encrypt bytes with the given public key (pure NTRU)."""
    encryptor = PureNTRUEncryptor(public_key)
    return encryptor.encrypt_bytes(data)


def decrypt_bytes_pure_ntru(encrypted: Dict, private_key: Dict) -> bytes:
    """Decrypt bytes with the given private key (pure NTRU)."""
    decryptor = PureNTRUDecryptor(private_key)
    return decryptor.decrypt_bytes(encrypted)


if __name__ == "__main__":
    print("Testing Pure NTRU Encryption (No AES Layer)")
    print("=" * 60)

    from .keygen import NTRUKeyGenerator

    print("\n1. Generating NTRU keys...")
    generator = NTRUKeyGenerator()
    public_key, private_key = generator.generate_keys()

    test_metadata = {
        "name": "John Doe",
        "phone": "1234567890",
        "age": 25,
        "email": "john.doe@example.com"
    }

    import json
    test_data = json.dumps(test_metadata).encode('utf-8')
    print(f"\n2. Test data: {test_data}")
    print(f"   Length: {len(test_data)} bytes")

    print("\n3. Encrypting with pure NTRU...")
    encryptor = PureNTRUEncryptor(public_key)
    encrypted = encryptor.encrypt_bytes(test_data)
    print(f"   Version: {encrypted['version']} (Pure NTRU)")
    print(f"   Number of polynomial chunks: {encrypted['num_chunks']}")
    print(f"   First ciphertext shape: {encrypted['ciphertexts'][0].shape}")

    print("\n4. Decrypting...")
    decryptor = PureNTRUDecryptor(private_key)
    decrypted = decryptor.decrypt_bytes(encrypted)
    print(f"   Decrypted: {decrypted}")

    print("\n5. Verification:")
    if decrypted == test_data:
        print("   SUCCESS: Decrypted data matches original!")
    else:
        print("   FAILED: Data mismatch!")
        print(f"   Original: {test_data}")
        print(f"   Decrypted: {decrypted}")

    print("\n" + "=" * 60)
    print("Testing Metadata Encryption")

    encrypted_meta = encryptor.encrypt_metadata(test_metadata)
    decrypted_meta = decryptor.decrypt_metadata(encrypted_meta)
    print(f"   Original: {test_metadata}")
    print(f"   Decrypted: {decrypted_meta}")

    if decrypted_meta == test_metadata:
        print("   SUCCESS: Metadata encryption/decryption works!")
    else:
        print("   FAILED: Metadata mismatch!")

    print("\n" + "=" * 60)
    print("Pure NTRU encryption test complete!")
