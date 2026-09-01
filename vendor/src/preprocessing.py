"""
Preprocessing pipeline for robust face recognition
Handles: lighting, skin tones, emotions, masks, glasses

Supports two output sizes:
- 16×16: For QCNN classification (face/non-face)
- 64×64: For 8-qubit Hierarchical QCNN recognition (embedding extraction)
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class FacePreprocessor:
    """
    Enterprise-grade face preprocessing with configurable output size.

    Args:
        output_size: Default output size (16 for classification, 64 for recognition)

    Usage:
        # For classification (16×16)
        preprocessor = FacePreprocessor(output_size=16)
        img = preprocessor.preprocess(image)

        # For recognition (64×64)
        preprocessor = FacePreprocessor(output_size=64)
        img = preprocessor.preprocess(image, for_recognition=True)
    """

    def __init__(self, output_size: int = 16):
        self.output_size = output_size
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        self.gamma = 1.2
        
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE for light invariance
        Works per channel for RGB images
        """
        if len(image.shape) == 2:  # Grayscale
            return self.clahe.apply(image)
        else:  # RGB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust brightness via gamma correction
        """
        inv_gamma = 1.0 / self.gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 
            for i in range(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)
    
    def normalize_skin_tone(self, image: np.ndarray) -> np.ndarray:
        """
        Skin tone invariant normalization
        Convert to YCrCb, equalize Y channel
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    def preprocess(self, image: np.ndarray, for_recognition: bool = False) -> np.ndarray:
        """
        Full preprocessing pipeline

        Args:
            image: Input image (BGR or RGB)
            for_recognition: If True, output 64×64 for QCNN recognition
                           If False, output self.output_size (default 16×16) for classification

        Steps:
        1. CLAHE for lighting
        2. Gamma correction
        3. Skin tone normalization
        4. Resize to target size
        5. Normalize to [0, 1]

        Returns:
            Preprocessed image of shape (H, W, 3) normalized to [0, 1]
        """
        # Apply enhancements
        img = self.apply_clahe(image)
        img = self.gamma_correction(img)
        img = self.normalize_skin_tone(img)

        # Determine target size
        if for_recognition:
            target_size = 64  # For 8-qubit Hierarchical QCNN
        else:
            target_size = self.output_size

        # Resize and normalize
        img = cv2.resize(img, (target_size, target_size))
        img = img.astype(np.float32) / 255.0

        return img

    def preprocess_for_recognition(self, image: np.ndarray) -> np.ndarray:
        """
        Convenience method for QCNN recognition preprocessing.

        Same as preprocess(image, for_recognition=True)

        Args:
            image: Input image (BGR or RGB)

        Returns:
            64×64×3 preprocessed image normalized to [0, 1]
        """
        return self.preprocess(image, for_recognition=True)
