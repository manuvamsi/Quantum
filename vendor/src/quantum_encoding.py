"""
Quantum encoding utilities
RYZ encoding for PCA-reduced features
"""

import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class QuantumEncoder:
    """
    Encode classical image data into quantum states
    Uses pre-trained PCA and scaler models for inference
    """

    def __init__(self, n_components: int = 10, models_dir: str = "models"):
        self.n_components = n_components
        # Always resolve models_dir relative to project root
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), models_dir)
        self.pca = None
        self.scaler = None

        # Try to load pre-trained models
        self._load_models()

    def _load_models(self):
        """Load pre-trained PCA and scaler models"""
        pca_path = os.path.join(self.models_dir, "pca_model.pkl")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")

        if os.path.exists(pca_path) and os.path.exists(scaler_path):
            self.pca = joblib.load(pca_path)
            self.scaler = joblib.load(scaler_path)
        else:
            print(f"Warning: PCA/scaler models not found in {self.models_dir}")

    def fit(self, X: np.ndarray):
        """
        Fit PCA and scaler on training data

        Args:
            X: (n_samples, 768) flattened 16×16×3 images
        """
        # Fit scaler first
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(X_scaled)

        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

    def reduce_dimensions(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Reduce dimensions using PCA (768 -> 10)

        Args:
            feature_vector: (768,) flattened 16×16×3 image

        Returns:
            (10,) PCA-reduced and scaled features
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("Models not loaded. Please load or fit models first.")

        # Reshape if needed
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Order: PCA first (768 -> 10), then StandardScaler (on 10 features)
        X_pca = self.pca.transform(feature_vector)
        X_scaled = self.scaler.transform(X_pca)

        return X_scaled[0]  # Return 1D array (10 scaled features)

    def encode_ry_rz(self, reduced_features: np.ndarray) -> np.ndarray:
        """
        Encode reduced features for quantum circuit (normalize to [0, π])

        Args:
            reduced_features: (10,) PCA-reduced features

        Returns:
            (10,) features normalized to [0, π] for RY/RZ gates
        """
        # FIXED normalization: clip to [-3, 3] range then scale to [0, π]
        # This ensures consistent normalization (same as training)
        features = np.clip(reduced_features, -3, 3)
        features = (features + 3) / 6 * np.pi

        return features

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform image data to quantum features

        Args:
            X: (n_samples, 768) flattened images

        Returns:
            (n_samples, 10) quantum features normalized to [0, π]
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("Models not loaded. Please load or fit models first.")

        # Order: PCA first (768 -> 10), then StandardScaler (on 10 features)
        X_pca = self.pca.transform(X)
        X_scaled = self.scaler.transform(X_pca)

        # FIXED normalization: clip to [-3, 3] range then scale to [0, π]
        X_scaled = np.clip(X_scaled, -3, 3)
        X_norm = (X_scaled + 3) / 6 * np.pi

        return X_norm

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform
        """
        self.fit(X)
        return self.transform(X)
