"""
Secure Metadata Storage with PQC-NTRU Encryption

Provides encrypted storage for user metadata (name, phone, etc.)
using NTRU + AES-256-GCM hybrid encryption.

Storage Structure:
PQC/encrypted_metadata/
└── {user_id}.meta.enc    ← JSON encrypted with NTRU + AES-GCM

Metadata Format (JSON, extensible):
{
    "name": "John Doe",
    "phone": "1234567890",
    "timestamp": "2024-01-15T10:30:00",
    "version": 1
    // Future fields can be added without code changes
}
"""

import os
import json
import pickle
from typing import Dict, Optional, List
from datetime import datetime

from .ntru import NTRUKeyGenerator, NTRUEncryptor, NTRUDecryptor
from .timing import TimingLogger, get_timing_logger


class SecureMetadataStorage:
    """
    PQC-encrypted metadata storage

    Stores user metadata (name, phone, future fields) encrypted with
    NTRU + AES-256-GCM. Links to ChromaDB via user_id.
    """

    def __init__(self, storage_dir: str = None, auto_init_keys: bool = True):
        """
        Initialize secure metadata storage

        Args:
            storage_dir: Directory for encrypted metadata
            auto_init_keys: Automatically generate keys if not exist
        """
        # Set up directory
        if storage_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.storage_dir = os.path.join(base_dir, "encrypted_metadata")
        else:
            self.storage_dir = storage_dir

        os.makedirs(self.storage_dir, exist_ok=True)

        # Initialize key manager
        self.key_generator = NTRUKeyGenerator()

        # Load or generate keys
        if auto_init_keys:
            self._init_keys()
        else:
            self.public_key = None
            self.private_key = None
            self.encryptor = None
            self.decryptor = None

    def _init_keys(self):
        """Initialize NTRU keys"""
        if self.key_generator.keys_exist():
            self.public_key, self.private_key = self.key_generator.load_keys()
        else:
            print("Generating NTRU keys for secure metadata storage...")
            self.public_key, self.private_key = self.key_generator.generate_and_save()

        self.encryptor = NTRUEncryptor(self.public_key)
        self.decryptor = NTRUDecryptor(self.private_key)

    def _get_metadata_path(self, user_id: str) -> str:
        """Get path to encrypted metadata file for a user"""
        return os.path.join(self.storage_dir, f"{user_id}.meta.enc")

    def store_metadata(self, user_id: str, metadata: Dict,
                       timing_logger: TimingLogger = None) -> str:
        """
        Store encrypted metadata for a user

        Args:
            user_id: User identifier (links to ChromaDB)
            metadata: Dict with name, phone, and any future fields

        Example metadata:
            {
                "name": "John Doe",
                "phone": "1234567890",
                // Future fields added here without code changes
            }

        Returns:
            Path to stored encrypted metadata
        """
        if self.encryptor is None:
            self._init_keys()

        logger = timing_logger or get_timing_logger()

        # Add timestamp and version if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
        if 'version' not in metadata:
            metadata['version'] = 1

        # Convert metadata dict to JSON bytes
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        metadata_bytes = metadata_json.encode('utf-8')

        # Encrypt using NTRU + AES-256-GCM
        with logger.time_encryption():
            encrypted = self._encrypt_bytes(metadata_bytes)

        # Save encrypted metadata
        meta_path = self._get_metadata_path(user_id)
        with open(meta_path, 'wb') as f:
            pickle.dump(encrypted, f)

        return meta_path

    def load_metadata(self, user_id: str,
                      timing_logger: TimingLogger = None) -> Optional[Dict]:
        """
        Load and decrypt metadata for a user

        Args:
            user_id: User identifier

        Returns:
            Decrypted metadata dict or None if not found
        """
        if self.decryptor is None:
            self._init_keys()

        logger = timing_logger or get_timing_logger()

        meta_path = self._get_metadata_path(user_id)

        if not os.path.exists(meta_path):
            return None

        # Load encrypted metadata
        with logger.time_retrieval_accumulate():
            with open(meta_path, 'rb') as f:
                encrypted = pickle.load(f)

        # Decrypt
        with logger.time_decryption_accumulate():
            metadata_bytes = self._decrypt_bytes(encrypted)

        if metadata_bytes is None:
            return None

        # Parse JSON
        try:
            metadata_json = metadata_bytes.decode('utf-8')
            return json.loads(metadata_json)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error parsing metadata for {user_id}: {e}")
            return None

    def update_metadata(self, user_id: str, updates: Dict,
                        timing_logger: TimingLogger = None) -> bool:
        """
        Update specific fields without re-registration

        Decrypts existing metadata, merges updates, re-encrypts.

        Args:
            user_id: User identifier
            updates: Dict with fields to update/add

        Returns:
            True if successful, False if user not found
        """
        # Load existing metadata
        existing = self.load_metadata(user_id, timing_logger)
        if existing is None:
            return False

        # Merge updates
        existing.update(updates)
        existing['timestamp'] = datetime.now().isoformat()  # Update timestamp

        # Re-encrypt and store
        self.store_metadata(user_id, existing, timing_logger)
        return True

    def delete_metadata(self, user_id: str) -> bool:
        """
        Delete metadata for a user

        Args:
            user_id: User to delete

        Returns:
            True if deleted, False if not found
        """
        meta_path = self._get_metadata_path(user_id)

        if os.path.exists(meta_path):
            os.remove(meta_path)
            return True
        return False

    def user_exists(self, user_id: str) -> bool:
        """Check if metadata exists for user"""
        return os.path.exists(self._get_metadata_path(user_id))

    def get_all_user_ids(self) -> List[str]:
        """Get list of all stored user IDs"""
        if not os.path.exists(self.storage_dir):
            return []

        user_ids = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.meta.enc'):
                user_id = filename[:-9]  # Remove '.meta.enc'
                user_ids.append(user_id)
        return user_ids

    def get_stats(self) -> Dict:
        """Get storage statistics"""
        user_ids = self.get_all_user_ids()

        return {
            'total_users': len(user_ids),
            'users': user_ids,
            'storage_dir': self.storage_dir
        }

    def _encrypt_bytes(self, data: bytes) -> Dict:
        """
        Encrypt bytes using Pure NTRU (no AES layer)

        Complete quantum NTRU-based encryption for metadata.

        Args:
            data: Bytes to encrypt

        Returns:
            Encrypted data dictionary (version 4 = Pure NTRU)
        """
        from .ntru.pure_ntru import PureNTRUEncryptor

        pure_encryptor = PureNTRUEncryptor(self.public_key)
        return pure_encryptor.encrypt_bytes(data)

    def _decrypt_bytes(self, encrypted: Dict) -> Optional[bytes]:
        """
        Decrypt bytes - supports both Pure NTRU (v4) and Hybrid (v3)

        Args:
            encrypted: Encrypted data dictionary

        Returns:
            Decrypted bytes or None if failed
        """
        version = encrypted.get('version', 1)

        try:
            if version >= 4:
                # Version 4+: Pure NTRU (no AES)
                return self._decrypt_bytes_pure_ntru(encrypted)
            else:
                # Version 3 and below: Hybrid (NTRU + AES-256-GCM)
                return self._decrypt_bytes_hybrid(encrypted)

        except Exception as e:
            print(f"Decryption error: {e}")
            return None

    def _decrypt_bytes_pure_ntru(self, encrypted: Dict) -> Optional[bytes]:
        """
        Decrypt using Pure NTRU (version 4+)

        Args:
            encrypted: Encrypted data dictionary

        Returns:
            Decrypted bytes
        """
        from .ntru.pure_ntru import PureNTRUDecryptor

        pure_decryptor = PureNTRUDecryptor(self.private_key)
        return pure_decryptor.decrypt_bytes(encrypted)

    def _decrypt_bytes_hybrid(self, encrypted: Dict) -> Optional[bytes]:
        """
        Decrypt using Hybrid mode (NTRU + AES-256-GCM) for version 3 and below

        Maintains backwards compatibility with existing encrypted data.

        Args:
            encrypted: Encrypted data dictionary

        Returns:
            Decrypted bytes or None if failed
        """
        import hashlib
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.exceptions import InvalidTag

        try:
            seed_coeffs = encrypted.get('seed_coeffs')
            aes_nonce = encrypted['aes_nonce']
            aes_ciphertext = encrypted['aes_ciphertext']
            aad = encrypted['aad']

            # Derive AES key from seed
            seed_bytes = seed_coeffs.tobytes()
            aes_key = hashlib.sha256(seed_bytes).digest()

            # Decrypt with AES-256-GCM
            aesgcm = AESGCM(aes_key)
            return aesgcm.decrypt(aes_nonce, aes_ciphertext, aad)

        except InvalidTag:
            print("Decryption failed: Data integrity check failed")
            return None
        except Exception as e:
            print(f"Hybrid decryption error: {e}")
            return None


# Global storage instance
_global_metadata_storage: Optional[SecureMetadataStorage] = None


def get_metadata_storage() -> SecureMetadataStorage:
    """Get or create global secure metadata storage"""
    global _global_metadata_storage
    if _global_metadata_storage is None:
        _global_metadata_storage = SecureMetadataStorage()
    return _global_metadata_storage


if __name__ == "__main__":
    # Test secure metadata storage
    print("Testing Secure Metadata Storage...")
    print("=" * 60)

    storage = SecureMetadataStorage()

    # Store test metadata
    print("\nStoring encrypted metadata...")
    test_metadata = {
        "name": "John Doe",
        "phone": "9876543210"
    }

    path = storage.store_metadata("john_doe", test_metadata)
    print(f"Stored at: {path}")

    # Load and verify
    print("\nLoading and decrypting metadata...")
    loaded = storage.load_metadata("john_doe")

    if loaded:
        print(f"Loaded metadata:")
        print(f"  Name: {loaded.get('name')}")
        print(f"  Phone: {loaded.get('phone')}")
        print(f"  Timestamp: {loaded.get('timestamp')}")
        print(f"  Version: {loaded.get('version')}")

        # Verify
        if loaded.get('name') == "John Doe" and loaded.get('phone') == "9876543210":
            print("\nSUCCESS: Metadata encryption/decryption works!")
        else:
            print("\nWARNING: Metadata mismatch")
    else:
        print("Failed to load metadata")

    # Test update
    print("\n" + "=" * 60)
    print("Testing metadata update...")

    success = storage.update_metadata("john_doe", {"email": "john@example.com"})
    if success:
        updated = storage.load_metadata("john_doe")
        print(f"Updated metadata:")
        print(f"  Name: {updated.get('name')}")
        print(f"  Phone: {updated.get('phone')}")
        print(f"  Email: {updated.get('email')}")
        print("SUCCESS: Update without re-registration works!")

    # Get stats
    print("\n" + "=" * 60)
    print("Storage stats:")
    stats = storage.get_stats()
    print(f"  Total users: {stats['total_users']}")
    print(f"  Users: {stats['users']}")
