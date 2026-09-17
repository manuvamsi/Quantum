"""Polynomial-ring operations used by NTRU."""

import numpy as np
from typing import Tuple, Optional


class PolynomialRing:
    """Arithmetic on polynomials for NTRU (dimension N, moduli q and p)."""

    def __init__(self, N: int = 509, q: int = 2048, p: int = 3):
        self.N = N
        self.q = q
        self.p = p

    def add(self, a: np.ndarray, b: np.ndarray, mod: Optional[int] = None) -> np.ndarray:
        """Add two polynomials, optionally reducing mod `mod`."""
        result = a + b
        if mod is not None:
            result = self._center_lift(result, mod)
        return result

    def subtract(self, a: np.ndarray, b: np.ndarray, mod: Optional[int] = None) -> np.ndarray:
        """Subtract two polynomials, optionally reducing mod `mod`."""
        result = a - b
        if mod is not None:
            result = self._center_lift(result, mod)
        return result

    def multiply(self, a: np.ndarray, b: np.ndarray, mod: Optional[int] = None) -> np.ndarray:
        """Multiply two polynomials in the ring, optionally reducing mod `mod`."""
        N = self.N

        a = self._pad_to_N(a)
        b = self._pad_to_N(b)

        result = np.zeros(N, dtype=np.int64)

        for i in range(N):
            for j in range(N):
                k = (i + j) % N
                result[k] += int(a[i]) * int(b[j])

        if mod is not None:
            result = self._center_lift(result, mod)

        return result

    def scalar_multiply(self, a: np.ndarray, scalar: int, mod: Optional[int] = None) -> np.ndarray:
        """Multiply a polynomial by a scalar, optionally reducing mod `mod`."""
        result = a * scalar
        if mod is not None:
            result = self._center_lift(result, mod)
        return result.astype(np.int64)

    def _center_lift(self, coeffs: np.ndarray, mod: int) -> np.ndarray:
        """Reduce coefficients into the range [-mod/2, mod/2)."""
        coeffs = coeffs % mod
        coeffs = np.where(coeffs > mod // 2, coeffs - mod, coeffs)
        return coeffs.astype(np.int64)

    def _pad_to_N(self, poly: np.ndarray) -> np.ndarray:
        """Pad or truncate a polynomial to length N."""
        if len(poly) < self.N:
            return np.pad(poly, (0, self.N - len(poly)), mode='constant')
        return poly[:self.N]

    def mod_reduce(self, poly: np.ndarray, mod: int) -> np.ndarray:
        """Reduce coefficients modulo `mod` (center-lifted)."""
        return self._center_lift(poly, mod)

    def inverse_mod_prime(self, poly: np.ndarray, mod: int) -> Optional[np.ndarray]:
        """Return the polynomial's inverse mod a prime, or None if none exists."""
        N = self.N
        poly = self._pad_to_N(poly)

        r0 = np.zeros(N + 1, dtype=np.int64)
        r0[N] = 1
        r0[0] = -1

        r1 = np.zeros(N + 1, dtype=np.int64)
        r1[:N] = poly

        s0 = np.zeros(N + 1, dtype=np.int64)
        s1 = np.zeros(N + 1, dtype=np.int64)
        s1[0] = 1

        while not self._is_zero(r1):
            q, r = self._poly_divmod(r0, r1, mod)
            r0, r1 = r1, r
            s0, s1 = s1, self._poly_sub_mod(s0, self._poly_mul_mod(q, s1, mod), mod)

        deg_r0 = self._degree(r0)
        if deg_r0 > 0:
            return None

        c = r0[0] % mod
        if c == 0:
            return None

        c_inv = pow(int(c), mod - 2, mod)
        result = (s0 * c_inv) % mod
        result = self._center_lift(result[:N], mod)

        return result

    def inverse_mod_power_of_2(self, poly: np.ndarray, q: int) -> Optional[np.ndarray]:
        """Return the polynomial's inverse mod a power of two, or None."""
        inv = self.inverse_mod_prime(poly, 2)
        if inv is None:
            return None

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
        """Divide polynomial a by b, returning (quotient, remainder)."""
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
        """Non-cyclic polynomial multiplication mod `mod`."""
        result = np.convolve(a, b)
        return self._center_lift(result, mod)

    def _poly_sub_mod(self, a: np.ndarray, b: np.ndarray, mod: int) -> np.ndarray:
        """Length-matched polynomial subtraction mod `mod`."""
        max_len = max(len(a), len(b))
        a_padded = np.pad(a, (0, max_len - len(a)))
        b_padded = np.pad(b, (0, max_len - len(b)))
        return self._center_lift(a_padded - b_padded, mod)

    def _degree(self, poly: np.ndarray) -> int:
        """Return the degree of a polynomial (-1 if zero)."""
        for i in range(len(poly) - 1, -1, -1):
            if poly[i] != 0:
                return i
        return -1

    def _is_zero(self, poly: np.ndarray) -> bool:
        """Return True if the polynomial is all zeros."""
        return np.all(poly == 0)

    def random_ternary(self, num_ones: int, num_neg_ones: int) -> np.ndarray:
        """Random polynomial with the given counts of +1 and -1 coefficients."""
        poly = np.zeros(self.N, dtype=np.int64)
        indices = np.random.permutation(self.N)

        poly[indices[:num_ones]] = 1
        poly[indices[num_ones:num_ones + num_neg_ones]] = -1

        return poly

    def random_small(self, bound: int = 1) -> np.ndarray:
        """Random polynomial with coefficients in [-bound, bound]."""
        return np.random.randint(-bound, bound + 1, size=self.N, dtype=np.int64)

    def from_bytes(self, data: bytes) -> np.ndarray:
        """Convert a byte string to polynomial coefficients."""
        coeffs = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        return self._pad_to_N(coeffs)

    def to_bytes(self, poly: np.ndarray) -> bytes:
        """Convert polynomial coefficients back to bytes."""
        coeffs = (poly % 256).astype(np.uint8)
        return coeffs.tobytes()
