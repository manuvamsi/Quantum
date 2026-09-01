"""
Polynomial Ring Operations for NTRU

Ring: R = Z[X]/(X^N - 1)
- Polynomials of degree at most N-1
- Coefficients are integers
- Multiplication is cyclic convolution (X^N = 1)

This implementation follows the mathematical notation from NTRU specification.
"""

import numpy as np
from typing import Tuple, Optional


class PolynomialRing:
    """
    Operations in the polynomial ring Z[X]/(X^N - 1)

    Parameters:
        N: Polynomial degree (ring dimension)
        q: Large modulus for public key operations
        p: Small modulus for message space
    """

    def __init__(self, N: int = 509, q: int = 2048, p: int = 3):
        self.N = N
        self.q = q
        self.p = p

    def add(self, a: np.ndarray, b: np.ndarray, mod: Optional[int] = None) -> np.ndarray:
        """
        Add two polynomials: (a + b) mod modulus

        Args:
            a: First polynomial coefficients
            b: Second polynomial coefficients
            mod: Optional modulus (default: no modular reduction)

        Returns:
            Sum polynomial coefficients
        """
        result = a + b
        if mod is not None:
            result = self._center_lift(result, mod)
        return result

    def subtract(self, a: np.ndarray, b: np.ndarray, mod: Optional[int] = None) -> np.ndarray:
        """
        Subtract two polynomials: (a - b) mod modulus
        """
        result = a - b
        if mod is not None:
            result = self._center_lift(result, mod)
        return result

    def multiply(self, a: np.ndarray, b: np.ndarray, mod: Optional[int] = None) -> np.ndarray:
        """
        Multiply two polynomials in R = Z[X]/(X^N - 1)
        Uses cyclic convolution: X^N = 1

        Args:
            a: First polynomial coefficients (length N)
            b: Second polynomial coefficients (length N)
            mod: Optional modulus

        Returns:
            Product polynomial coefficients (length N)
        """
        N = self.N

        # Pad to length N if needed
        a = self._pad_to_N(a)
        b = self._pad_to_N(b)

        # Cyclic convolution using FFT for efficiency
        # c[k] = sum(a[i] * b[(k-i) mod N] for i in range(N))
        result = np.zeros(N, dtype=np.int64)

        for i in range(N):
            for j in range(N):
                k = (i + j) % N
                result[k] += int(a[i]) * int(b[j])

        if mod is not None:
            result = self._center_lift(result, mod)

        return result

    def scalar_multiply(self, a: np.ndarray, scalar: int, mod: Optional[int] = None) -> np.ndarray:
        """
        Multiply polynomial by scalar
        """
        result = a * scalar
        if mod is not None:
            result = self._center_lift(result, mod)
        return result.astype(np.int64)

    def _center_lift(self, coeffs: np.ndarray, mod: int) -> np.ndarray:
        """
        Center-lift coefficients to range [-mod/2, mod/2)
        This is standard in NTRU to keep coefficients small
        """
        coeffs = coeffs % mod
        coeffs = np.where(coeffs > mod // 2, coeffs - mod, coeffs)
        return coeffs.astype(np.int64)

    def _pad_to_N(self, poly: np.ndarray) -> np.ndarray:
        """Pad polynomial to length N"""
        if len(poly) < self.N:
            return np.pad(poly, (0, self.N - len(poly)), mode='constant')
        return poly[:self.N]

    def mod_reduce(self, poly: np.ndarray, mod: int) -> np.ndarray:
        """
        Reduce polynomial coefficients modulo mod with center-lift
        """
        return self._center_lift(poly, mod)

    def inverse_mod_prime(self, poly: np.ndarray, mod: int) -> Optional[np.ndarray]:
        """
        Compute multiplicative inverse of polynomial modulo a prime
        Uses extended Euclidean algorithm in polynomial ring

        Args:
            poly: Polynomial to invert
            mod: Prime modulus

        Returns:
            Inverse polynomial such that poly * inverse = 1 (mod X^N-1, mod)
            Returns None if inverse doesn't exist
        """
        N = self.N
        poly = self._pad_to_N(poly)

        # Extended Euclidean algorithm for polynomials
        # We need to find inv such that poly * inv = 1 (mod X^N - 1, mod)

        # Start with r0 = X^N - 1, r1 = poly
        r0 = np.zeros(N + 1, dtype=np.int64)
        r0[N] = 1
        r0[0] = -1  # X^N - 1

        r1 = np.zeros(N + 1, dtype=np.int64)
        r1[:N] = poly

        s0 = np.zeros(N + 1, dtype=np.int64)
        s1 = np.zeros(N + 1, dtype=np.int64)
        s1[0] = 1

        while not self._is_zero(r1):
            q, r = self._poly_divmod(r0, r1, mod)
            r0, r1 = r1, r
            s0, s1 = s1, self._poly_sub_mod(s0, self._poly_mul_mod(q, s1, mod), mod)

        # Check if GCD is a constant (invertible)
        deg_r0 = self._degree(r0)
        if deg_r0 > 0:
            return None  # No inverse exists

        # Normalize by the constant
        c = r0[0] % mod
        if c == 0:
            return None

        c_inv = pow(int(c), mod - 2, mod)  # Fermat's little theorem
        result = (s0 * c_inv) % mod
        result = self._center_lift(result[:N], mod)

        return result

    def inverse_mod_power_of_2(self, poly: np.ndarray, q: int) -> Optional[np.ndarray]:
        """
        Compute multiplicative inverse modulo a power of 2 using Newton's method

        Args:
            poly: Polynomial to invert
            q: Power of 2 modulus

        Returns:
            Inverse polynomial
        """
        # First compute inverse mod 2
        inv = self.inverse_mod_prime(poly, 2)
        if inv is None:
            return None

        # Newton iteration: inv = inv * (2 - poly * inv) mod q
        # Double the precision with each iteration
        mod = 2
        while mod < q:
            mod = min(mod * mod, q)
            prod = self.multiply(poly, inv, mod)
            two_minus = np.zeros(self.N, dtype=np.int64)
            two_minus[0] = 2
            diff = self.subtract(two_minus, prod, mod)
            inv = self.multiply(inv, diff, mod)

        return inv

    def _poly_divmod(self, a: np.ndarray, b: np.ndarray, mod: int) -> Tuple[np.ndarray, np.ndarray]:
        """Polynomial division with remainder"""
        a = a.copy().astype(np.int64)
        b = b.copy().astype(np.int64)

        deg_a = self._degree(a)
        deg_b = self._degree(b)

        if deg_b < 0:
            raise ValueError("Division by zero polynomial")

        q = np.zeros(len(a), dtype=np.int64)

        b_lead_inv = pow(int(b[deg_b]), mod - 2, mod)

        while deg_a >= deg_b:
            coeff = (a[deg_a] * b_lead_inv) % mod
            shift = deg_a - deg_b
            q[shift] = coeff

            for i in range(deg_b + 1):
                a[i + shift] = (a[i + shift] - coeff * b[i]) % mod

            deg_a = self._degree(a)

        return self._center_lift(q, mod), self._center_lift(a, mod)

    def _poly_mul_mod(self, a: np.ndarray, b: np.ndarray, mod: int) -> np.ndarray:
        """Simple polynomial multiplication (not cyclic) mod prime"""
        result = np.convolve(a, b)
        return self._center_lift(result, mod)

    def _poly_sub_mod(self, a: np.ndarray, b: np.ndarray, mod: int) -> np.ndarray:
        """Polynomial subtraction with length matching"""
        max_len = max(len(a), len(b))
        a_padded = np.pad(a, (0, max_len - len(a)))
        b_padded = np.pad(b, (0, max_len - len(b)))
        return self._center_lift(a_padded - b_padded, mod)

    def _degree(self, poly: np.ndarray) -> int:
        """Find degree of polynomial"""
        for i in range(len(poly) - 1, -1, -1):
            if poly[i] != 0:
                return i
        return -1

    def _is_zero(self, poly: np.ndarray) -> bool:
        """Check if polynomial is zero"""
        return np.all(poly == 0)

    def random_ternary(self, num_ones: int, num_neg_ones: int) -> np.ndarray:
        """
        Generate random ternary polynomial with specified number of +1s and -1s
        Used for generating f, g, and r in NTRU

        Args:
            num_ones: Number of +1 coefficients
            num_neg_ones: Number of -1 coefficients

        Returns:
            Ternary polynomial (coefficients in {-1, 0, 1})
        """
        poly = np.zeros(self.N, dtype=np.int64)
        indices = np.random.permutation(self.N)

        poly[indices[:num_ones]] = 1
        poly[indices[num_ones:num_ones + num_neg_ones]] = -1

        return poly

    def random_small(self, bound: int = 1) -> np.ndarray:
        """
        Generate random polynomial with small coefficients

        Args:
            bound: Coefficients will be in range [-bound, bound]
        """
        return np.random.randint(-bound, bound + 1, size=self.N, dtype=np.int64)

    def from_bytes(self, data: bytes) -> np.ndarray:
        """
        Convert byte array to polynomial coefficients
        Used for encoding embeddings as polynomials
        """
        # Convert bytes to integers and pad to N
        coeffs = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        return self._pad_to_N(coeffs)

    def to_bytes(self, poly: np.ndarray) -> bytes:
        """
        Convert polynomial coefficients back to bytes
        """
        # Ensure coefficients are positive for byte conversion
        coeffs = (poly % 256).astype(np.uint8)
        return coeffs.tobytes()
