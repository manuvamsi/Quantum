"""NTRU key generation."""

import numpy as np
import os
import pickle
from typing import Dict, Tuple, Optional
from .polynomial_ring import PolynomialRing


class NTRUKeyGenerator:
    """Generates and stores NTRU public/private key pairs."""

    DEFAULT_N = 509
    DEFAULT_P = 3
    DEFAULT_Q = 2048

    def __init__(self, N: int = None, p: int = None, q: int = None, keys_dir: str = None):
        """Set up parameters and the keys directory."""
        self.N = N or self.DEFAULT_N
        self.p = p or self.DEFAULT_P
        self.q = q or self.DEFAULT_Q

        self.df = self.N // 3
        self.dg = self.N // 3

        self.ring = PolynomialRing(self.N, self.q, self.p)

        if keys_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.keys_dir = os.path.join(base_dir, "keys")
        else:
            self.keys_dir = keys_dir

        os.makedirs(self.keys_dir, exist_ok=True)

    def generate_keys(self) -> Tuple[Dict, Dict]:
        """Generate an NTRU public/private key pair."""
        print(f"Generating NTRU keys with N={self.N}, p={self.p}, q={self.q}...")

        f, Fp, Fq = self._generate_f()

        g = self.ring.random_ternary(self.dg, self.dg)

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
        """Generate a private polynomial f that is invertible mod p and mod q, with its inverses."""
        max_attempts = 100

        for attempt in range(max_attempts):
            f = self.ring.random_ternary(self.df, self.df)

            if np.sum(f) == 0:
                f[0] = 1

            try:
                Fp = self.ring.inverse_mod_prime(f, self.p)
                if Fp is None:
                    continue

                Fq = self.ring.inverse_mod_power_of_2(f, self.q)
                if Fq is None:
                    continue

                check_p = self.ring.multiply(f, Fp, self.p)
                check_q = self.ring.multiply(f, Fq, self.q)

                is_identity_p = (check_p[0] == 1) and np.all(check_p[1:] == 0)
                is_identity_q = (check_q[0] == 1) and np.all(check_q[1:] == 0)

                if is_identity_p and is_identity_q:
                    return f, Fp, Fq

            except Exception:
                continue

        raise RuntimeError(f"Failed to generate invertible f after {max_attempts} attempts")

    def save_keys(self, public_key: Dict, private_key: Dict) -> Tuple[str, str]:
        """Save the key pair to disk and return the two file paths."""
        pub_path = os.path.join(self.keys_dir, "public_key.pkl")
        priv_path = os.path.join(self.keys_dir, "private_key.pkl")

        with open(pub_path, 'wb') as f:
            pickle.dump(public_key, f)

        with open(priv_path, 'wb') as f:
            pickle.dump(private_key, f)

        gitignore_path = os.path.join(self.keys_dir, ".gitignore")
        with open(gitignore_path, 'w') as f:
            f.write("# Never commit cryptographic keys\n")
            f.write("*.pkl\n")
            f.write("private_key.*\n")

        print(f"Public key saved to: {pub_path}")
        print(f"Private key saved to: {priv_path}")

        return pub_path, priv_path

    def load_keys(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load the key pair from disk (returns None for a missing key)."""
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
        """Return True if both key files are present."""
        pub_path = os.path.join(self.keys_dir, "public_key.pkl")
        priv_path = os.path.join(self.keys_dir, "private_key.pkl")
        return os.path.exists(pub_path) and os.path.exists(priv_path)

    def generate_and_save(self, force: bool = False) -> Tuple[Dict, Dict]:
        """Load existing keys, or generate and save a new pair (force=True always regenerates)."""
        if not force and self.keys_exist():
            print("Keys already exist. Loading existing keys...")
            return self.load_keys()

        public_key, private_key = self.generate_keys()
        self.save_keys(public_key, private_key)
        return public_key, private_key


def setup_keys(force: bool = False) -> Tuple[Dict, Dict]:
    """Generate or load NTRU keys."""
    generator = NTRUKeyGenerator()
    return generator.generate_and_save(force=force)


if __name__ == "__main__":
    print("Testing NTRU Key Generation...")
    print("=" * 60)

    generator = NTRUKeyGenerator()
    public_key, private_key = generator.generate_keys()

    print(f"\nPublic key h shape: {public_key['h'].shape}")
    print(f"Private key f shape: {private_key['f'].shape}")
    print(f"Private key Fp shape: {private_key['Fp'].shape}")

    generator.save_keys(public_key, private_key)

    loaded_pub, loaded_priv = generator.load_keys()
    print(f"\nKeys loaded successfully: {loaded_pub is not None and loaded_priv is not None}")
