"""
Pure NTRU Encryption (No AES Layer)

This module implements complete quantum NTRU-based encryption for metadata,
without using any classical symmetric cipher (AES).

Encryption Flow:
1. Convert bytes to ternary polynomial coefficients (base-3 encoding)
2. Chunk data into polynomial-sized blocks
3. Encrypt each polynomial: c = r * h + m (mod q)

Decryption Flow:
1. Decrypt each polynomial: a = f * c, then m = Fp * a (mod p)
2. Convert ternary coefficients back to bytes
3. Verify integrity via embedded checksum

Message Encoding:
- Each byte (0-255) → 6 ternary digits (base-3)
- N=509 coefficients per polynomial → ~84 bytes per polynomial
- Larger data is chunked across multiple polynomials

Security:
- Full post-quantum security from NTRU lattice problem
- Integrity via embedded checksum in payload
- No classical symmetric cipher dependency

Parameters (default):
- N = 509 (polynomial degree)
- p = 3 (small modulus, message space)
- q = 2048 (large modulus)
"""

import numpy as np
from typing import Dict, Optional, List
from .polynomial_ring import PolynomialRing


class PureNTRUEncryptor:
    """
    Pure NTRU Encryption without AES layer

    Encrypts data directly using NTRU polynomial encryption.
    For data larger than polynomial capacity, uses chunking.
    """

    def __init__(self, public_key: Dict):
        """
        Initialize encryptor with NTRU public key

        Args:
            public_key: Dictionary containing {'h', 'N', 'p', 'q'}
        """
        self.h = public_key['h']
        self.N = public_key['N']
        self.p = public_key['p']
        self.q = public_key['q']

        # Derived parameters
        self.dr = self.N // 3  # Number of 1s in blinding polynomial r

        # Initialize polynomial ring
        self.ring = PolynomialRing(self.N, self.q, self.p)

        # Capacity calculation:
        # Each byte (0-255) needs ceil(log3(256)) = 6 ternary digits
        # 509 coefficients / 6 ≈ 84 bytes per polynomial
        # Reserve some space for length and checksum
        self.bytes_per_poly = (self.N - 18) // 6  # ~81 bytes

    def _bytes_to_ternary(self, data: bytes) -> np.ndarray:
        """
        Convert bytes to ternary polynomial coefficients using base-3 encoding

        Each byte (0-255) is converted to 6 ternary digits {-1, 0, 1}

        Args:
            data: Bytes to convert

        Returns:
            Array of ternary coefficients
        """
        coeffs = []
        for byte in data:
            # Convert byte to base-3 (6 digits)
            val = byte
            for _ in range(6):
                digit = val % 3
                coeffs.append(digit - 1)  # Map 0,1,2 → -1,0,1
                val //= 3
        return np.array(coeffs, dtype=np.int64)

    def _ternary_to_bytes(self, coeffs: np.ndarray) -> bytes:
        """
        Convert ternary coefficients back to bytes

        Args:
            coeffs: Array of ternary coefficients {-1, 0, 1}

        Returns:
            Reconstructed bytes
        """
        result = []
        for i in range(0, len(coeffs), 6):
            chunk = coeffs[i:i+6]
            if len(chunk) < 6:
                break
            # Map -1,0,1 → 0,1,2 and convert from base-3
            val = 0
            for j in range(5, -1, -1):
                coeff_val = int(chunk[j])
                # Ensure coefficient is in {-1, 0, 1}
                if coeff_val < -1:
                    coeff_val = -1
                elif coeff_val > 1:
                    coeff_val = 1
                val = val * 3 + (coeff_val + 1)
            result.append(val % 256)
        return bytes(result)

    def encrypt_polynomial(self, message_poly: np.ndarray) -> np.ndarray:
        """
        NTRU encrypt a single message polynomial

        Encryption: c = r * h + m (mod q)

        Args:
            message_poly: Message polynomial (ternary coefficients)

        Returns:
            Ciphertext polynomial
        """
        # Generate random blinding polynomial r
        r = self.ring.random_ternary(self.dr, self.dr)

        # Compute r * h (mod q)
        r_h = self.ring.multiply(r, self.h, self.q)

        # Ensure message_poly is padded to N
        message_padded = self.ring._pad_to_N(message_poly)

        # Compute ciphertext: c = r * h + m (mod q)
        ciphertext = self.ring.add(r_h, message_padded, self.q)

        return ciphertext

    def encrypt_bytes(self, data: bytes) -> Dict:
        """
        Encrypt arbitrary bytes using pure NTRU

        Chunks data into polynomials and encrypts each.
        Adds length prefix and checksum for integrity verification.

        Args:
            data: Bytes to encrypt

        Returns:
            Dictionary containing encrypted data and metadata
        """
        # Compute checksum for integrity verification
        checksum = self._compute_checksum(data)

        # Create header: [length_low, length_high, checksum]
        length = len(data)
        header = bytes([
            length & 0xFF,           # Low byte of length
            (length >> 8) & 0xFF,    # High byte of length
            checksum                  # Checksum byte
        ])

        # Combine header and data
        full_data = header + data

        # Convert to ternary coefficients
        ternary = self._bytes_to_ternary(full_data)

        # Chunk into polynomials and encrypt
        ciphertexts = []
        for i in range(0, len(ternary), self.N):
            chunk = ternary[i:i + self.N]
            # Pad to N if needed
            if len(chunk) < self.N:
                chunk = np.pad(chunk, (0, self.N - len(chunk)), mode='constant')
            # Encrypt this polynomial
            ciphertext = self.encrypt_polynomial(chunk)
            ciphertexts.append(ciphertext)

        return {
            'ciphertexts': ciphertexts,
            'num_chunks': len(ciphertexts),
            'original_length': length,
            'N': self.N,
            'p': self.p,
            'q': self.q,
            'version': 4  # Version 4 = Pure NTRU
        }

    def encrypt_metadata(self, metadata_dict: Dict) -> Dict:
        """
        Encrypt a metadata dictionary using pure NTRU

        Args:
            metadata_dict: Dictionary with metadata fields

        Returns:
            Encrypted metadata dictionary
        """
        import json

        # Convert to JSON bytes
        metadata_json = json.dumps(metadata_dict, ensure_ascii=False)
        metadata_bytes = metadata_json.encode('utf-8')

        return self.encrypt_bytes(metadata_bytes)

    def _compute_checksum(self, data: bytes) -> int:
        """
        Compute a simple checksum for integrity verification

        Uses XOR-based checksum for better distribution than sum

        Args:
            data: Bytes to checksum

        Returns:
            Single byte checksum
        """
        checksum = 0
        for i, byte in enumerate(data):
            # XOR with rotation for better mixing
            checksum ^= ((byte << (i % 8)) | (byte >> (8 - i % 8))) & 0xFF
        return checksum


class PureNTRUDecryptor:
    """
    Pure NTRU Decryption without AES layer

    Decrypts data encrypted with PureNTRUEncryptor.
    """

    def __init__(self, private_key: Dict):
        """
        Initialize decryptor with NTRU private key

        Args:
            private_key: Dictionary containing {'f', 'Fp', 'N', 'p', 'q'}
        """
        self.f = private_key['f']
        self.Fp = private_key['Fp']
        self.N = private_key['N']
        self.p = private_key['p']
        self.q = private_key['q']

        # Initialize polynomial ring
        self.ring = PolynomialRing(self.N, self.q, self.p)

    def decrypt_polynomial(self, ciphertext: np.ndarray) -> np.ndarray:
        """
        NTRU decrypt a single ciphertext polynomial

        Decryption:
        1. a = f * c (mod q)
        2. a = center_lift(a, p)
        3. m = Fp * a (mod p)

        Args:
            ciphertext: Ciphertext polynomial

        Returns:
            Decrypted message polynomial (ternary)
        """
        # Step 1: a = f * c (mod q)
        a = self.ring.multiply(self.f, ciphertext, self.q)

        # Step 2: Center-lift and reduce mod p
        a = self.ring.mod_reduce(a, self.p)

        # Step 3: m = Fp * a (mod p)
        message = self.ring.multiply(self.Fp, a, self.p)

        return message

    def decrypt_bytes(self, encrypted: Dict) -> Optional[bytes]:
        """
        Decrypt encrypted bytes from pure NTRU format

        Args:
            encrypted: Dictionary from encrypt_bytes()

        Returns:
            Decrypted bytes or None if decryption fails
        """
        ciphertexts = encrypted['ciphertexts']
        original_length = encrypted.get('original_length', 0)

        # Decrypt all chunks
        all_coeffs = []
        for ciphertext in ciphertexts:
            message = self.decrypt_polynomial(ciphertext)
            all_coeffs.extend(message.tolist())

        # Convert ternary to bytes
        coeffs = np.array(all_coeffs, dtype=np.int64)
        decrypted = self._ternary_to_bytes(coeffs)

        # Extract header (length and checksum)
        if len(decrypted) < 3:
            raise ValueError("Decrypted data too short - missing header")

        length = decrypted[0] | (decrypted[1] << 8)
        stored_checksum = decrypted[2]
        data = decrypted[3:3 + length]

        # Verify length
        if len(data) != length:
            raise ValueError(f"Length mismatch: expected {length}, got {len(data)}")

        # Verify integrity
        computed_checksum = self._compute_checksum(data)
        if computed_checksum != stored_checksum:
            raise ValueError(
                f"Data integrity check failed: "
                f"expected checksum {stored_checksum}, got {computed_checksum}"
            )

        return bytes(data)

    def decrypt_metadata(self, encrypted: Dict) -> Optional[Dict]:
        """
        Decrypt encrypted metadata dictionary

        Args:
            encrypted: Dictionary from encrypt_metadata()

        Returns:
            Decrypted metadata dictionary or None
        """
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
        """
        Convert ternary coefficients back to bytes

        Args:
            coeffs: Array of ternary coefficients {-1, 0, 1}

        Returns:
            Reconstructed bytes
        """
        result = []
        for i in range(0, len(coeffs), 6):
            chunk = coeffs[i:i+6]
            if len(chunk) < 6:
                break
            # Map -1,0,1 → 0,1,2 and convert from base-3
            val = 0
            for j in range(5, -1, -1):
                coeff_val = int(chunk[j])
                # Ensure coefficient is in {-1, 0, 1}
                if coeff_val < -1:
                    coeff_val = -1
                elif coeff_val > 1:
                    coeff_val = 1
                val = val * 3 + (coeff_val + 1)
            result.append(val % 256)
        return bytes(result)

    def _compute_checksum(self, data: bytes) -> int:
        """
        Compute checksum matching the encryptor's algorithm

        Args:
            data: Bytes to checksum

        Returns:
            Single byte checksum
        """
        checksum = 0
        for i, byte in enumerate(data):
            checksum ^= ((byte << (i % 8)) | (byte >> (8 - i % 8))) & 0xFF
        return checksum


def encrypt_bytes_pure_ntru(data: bytes, public_key: Dict) -> Dict:
    """
    Convenience function to encrypt bytes using pure NTRU

    Args:
        data: Bytes to encrypt
        public_key: NTRU public key

    Returns:
        Encrypted data dictionary
    """
    encryptor = PureNTRUEncryptor(public_key)
    return encryptor.encrypt_bytes(data)


def decrypt_bytes_pure_ntru(encrypted: Dict, private_key: Dict) -> bytes:
    """
    Convenience function to decrypt bytes using pure NTRU

    Args:
        encrypted: Encrypted data dictionary
        private_key: NTRU private key

    Returns:
        Decrypted bytes
    """
    decryptor = PureNTRUDecryptor(private_key)
    return decryptor.decrypt_bytes(encrypted)


if __name__ == "__main__":
    # Test pure NTRU encryption/decryption
    print("Testing Pure NTRU Encryption (No AES Layer)")
    print("=" * 60)

    from .keygen import NTRUKeyGenerator

    # Generate keys
    print("\n1. Generating NTRU keys...")
    generator = NTRUKeyGenerator()
    public_key, private_key = generator.generate_keys()

    # Test data
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

    # Encrypt
    print("\n3. Encrypting with pure NTRU...")
    encryptor = PureNTRUEncryptor(public_key)
    encrypted = encryptor.encrypt_bytes(test_data)
    print(f"   Version: {encrypted['version']} (Pure NTRU)")
    print(f"   Number of polynomial chunks: {encrypted['num_chunks']}")
    print(f"   First ciphertext shape: {encrypted['ciphertexts'][0].shape}")

    # Decrypt
    print("\n4. Decrypting...")
    decryptor = PureNTRUDecryptor(private_key)
    decrypted = decryptor.decrypt_bytes(encrypted)
    print(f"   Decrypted: {decrypted}")

    # Verify
    print("\n5. Verification:")
    if decrypted == test_data:
        print("   SUCCESS: Decrypted data matches original!")
    else:
        print("   FAILED: Data mismatch!")
        print(f"   Original: {test_data}")
        print(f"   Decrypted: {decrypted}")

    # Test metadata encryption
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
