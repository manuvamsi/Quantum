"""
NTRU Key Generation

Key Generation Algorithm:
1. Choose random ternary polynomials f and g
2. Ensure f is invertible mod p and mod q
3. Compute Fp = f^(-1) mod p
4. Compute Fq = f^(-1) mod q
5. Public key: h = Fq * g (mod q)
6. Private key: (f, Fp)

Parameters (Standard Secure):
- N = 509 (polynomial degree)
- p = 3 (small modulus)
- q = 2048 (large modulus)
- df = 509//3 (number of 1s and -1s in f)
- dg = 509//3 (number of 1s and -1s in g)
"""

import numpy as np
import os
import pickle
from typing import Dict, Tuple, Optional
from .polynomial_ring import PolynomialRing


class NTRUKeyGenerator:
    """
    NTRU Key Pair Generator

    Generates public/private key pairs for NTRU encryption.
    Keys are stored in PQC/keys/ directory.
    """

    # Standard secure parameters
    DEFAULT_N = 509
    DEFAULT_P = 3
    DEFAULT_Q = 2048

    def __init__(self, N: int = None, p: int = None, q: int = None, keys_dir: str = None):
        """
        Initialize key generator with NTRU parameters

        Args:
            N: Polynomial degree (default: 509)
            p: Small modulus (default: 3)
            q: Large modulus (default: 2048)
            keys_dir: Directory to store keys
        """
        self.N = N or self.DEFAULT_N
        self.p = p or self.DEFAULT_P
        self.q = q or self.DEFAULT_Q

        # Derived parameters
        self.df = self.N // 3  # Number of 1s in f
        self.dg = self.N // 3  # Number of 1s in g

        # Initialize polynomial ring
        self.ring = PolynomialRing(self.N, self.q, self.p)

        # Keys directory
        if keys_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.keys_dir = os.path.join(base_dir, "keys")
        else:
            self.keys_dir = keys_dir

        os.makedirs(self.keys_dir, exist_ok=True)

    def generate_keys(self) -> Tuple[Dict, Dict]:
        """
        Generate NTRU public/private key pair

        Returns:
            (public_key, private_key) dictionaries containing:
            - public_key: {'h': polynomial, 'N': int, 'p': int, 'q': int}
            - private_key: {'f': polynomial, 'Fp': polynomial, 'N': int, 'p': int, 'q': int}
        """
        print(f"Generating NTRU keys with N={self.N}, p={self.p}, q={self.q}...")

        # Generate f with invertible property
        f, Fp, Fq = self._generate_f()

        # Generate g
        g = self.ring.random_ternary(self.dg, self.dg)

        # Compute public key: h = p * Fq * g (mod q)
        # Note: We multiply by p to ensure decryption works correctly
        Fq_g = self.ring.multiply(Fq, g, self.q)
        h = self.ring.scalar_multiply(Fq_g, self.p, self.q)

        public_key = {
            'h': h,
            'N': self.N,
            'p': self.p,
            'q': self.q
        }

        private_key = {
            'f': f,
            'Fp': Fp,
            'N': self.N,
            'p': self.p,
            'q': self.q
        }

        print("Key generation complete.")
        return public_key, private_key

    def _generate_f(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate private polynomial f that is invertible mod p and mod q

        Returns:
            (f, Fp, Fq) where:
            - f: Private polynomial
            - Fp: Inverse of f mod p
            - Fq: Inverse of f mod q
        """
        max_attempts = 100

        for attempt in range(max_attempts):
            # Generate ternary f with df ones and df negative ones
            # f must have form 1 + p*F for some F to ensure invertibility mod p
            f = self.ring.random_ternary(self.df, self.df)

            # Ensure f(1) != 0 for better chance of invertibility
            if np.sum(f) == 0:
                f[0] = 1

            # Try to compute inverses
            try:
                Fp = self.ring.inverse_mod_prime(f, self.p)
                if Fp is None:
                    continue

                Fq = self.ring.inverse_mod_power_of_2(f, self.q)
                if Fq is None:
                    continue

                # Verify inverses
                check_p = self.ring.multiply(f, Fp, self.p)
                check_q = self.ring.multiply(f, Fq, self.q)

                # Check if both are identity (1, 0, 0, ..., 0)
                is_identity_p = (check_p[0] == 1) and np.all(check_p[1:] == 0)
                is_identity_q = (check_q[0] == 1) and np.all(check_q[1:] == 0)

                if is_identity_p and is_identity_q:
                    return f, Fp, Fq

            except Exception:
                continue

        raise RuntimeError(f"Failed to generate invertible f after {max_attempts} attempts")

    def save_keys(self, public_key: Dict, private_key: Dict) -> Tuple[str, str]:
        """
        Save keys to files

        Args:
            public_key: Public key dictionary
            private_key: Private key dictionary

        Returns:
            (public_key_path, private_key_path)
        """
        pub_path = os.path.join(self.keys_dir, "public_key.pkl")
        priv_path = os.path.join(self.keys_dir, "private_key.pkl")

        with open(pub_path, 'wb') as f:
            pickle.dump(public_key, f)

        with open(priv_path, 'wb') as f:
            pickle.dump(private_key, f)

        # Create .gitignore to prevent committing keys
        gitignore_path = os.path.join(self.keys_dir, ".gitignore")
        with open(gitignore_path, 'w') as f:
            f.write("# Never commit cryptographic keys\n")
            f.write("*.pkl\n")
            f.write("private_key.*\n")

        print(f"Public key saved to: {pub_path}")
        print(f"Private key saved to: {priv_path}")

        return pub_path, priv_path

    def load_keys(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Load keys from files

        Returns:
            (public_key, private_key) or (None, None) if not found
        """
        pub_path = os.path.join(self.keys_dir, "public_key.pkl")
        priv_path = os.path.join(self.keys_dir, "private_key.pkl")

        public_key = None
        private_key = None

        if os.path.exists(pub_path):
            with open(pub_path, 'rb') as f:
                public_key = pickle.load(f)

        if os.path.exists(priv_path):
            with open(priv_path, 'rb') as f:
                private_key = pickle.load(f)

        return public_key, private_key

    def keys_exist(self) -> bool:
        """Check if keys already exist"""
        pub_path = os.path.join(self.keys_dir, "public_key.pkl")
        priv_path = os.path.join(self.keys_dir, "private_key.pkl")
        return os.path.exists(pub_path) and os.path.exists(priv_path)

    def generate_and_save(self, force: bool = False) -> Tuple[Dict, Dict]:
        """
        Generate keys and save to files

        Args:
            force: If True, regenerate even if keys exist

        Returns:
            (public_key, private_key)
        """
        if not force and self.keys_exist():
            print("Keys already exist. Loading existing keys...")
            return self.load_keys()

        public_key, private_key = self.generate_keys()
        self.save_keys(public_key, private_key)
        return public_key, private_key


def setup_keys(force: bool = False) -> Tuple[Dict, Dict]:
    """
    Convenience function to setup NTRU keys

    Args:
        force: If True, regenerate keys even if they exist

    Returns:
        (public_key, private_key)
    """
    generator = NTRUKeyGenerator()
    return generator.generate_and_save(force=force)


if __name__ == "__main__":
    # Test key generation
    print("Testing NTRU Key Generation...")
    print("=" * 60)

    generator = NTRUKeyGenerator()
    public_key, private_key = generator.generate_keys()

    print(f"\nPublic key h shape: {public_key['h'].shape}")
    print(f"Private key f shape: {private_key['f'].shape}")
    print(f"Private key Fp shape: {private_key['Fp'].shape}")

    # Save keys
    generator.save_keys(public_key, private_key)

    # Verify by loading
    loaded_pub, loaded_priv = generator.load_keys()
    print(f"\nKeys loaded successfully: {loaded_pub is not None and loaded_priv is not None}")
