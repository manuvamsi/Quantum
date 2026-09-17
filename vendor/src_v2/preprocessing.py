

import cv2
import numpy as np
from typing import Tuple, Optional


class FacePreprocessor:
 

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

    def preprocess(self, image: np.ndarray, for_recognition: bool = False,
                   skip_resize: bool = False) -> np.ndarray:
       
        # Apply enhancements
        img = self.apply_clahe(image)
        img = self.gamma_correction(img)
        img = self.normalize_skin_tone(img)

        # Resize unless skipped (already resized)
        if not skip_resize:
            if for_recognition:
                target_size = 64  # For 8-qubit Hierarchical QCNN
            else:
                target_size = self.output_size
            img = cv2.resize(img, (target_size, target_size))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def preprocess_for_recognition(self, image: np.ndarray) -> np.ndarray:
     
       
        return self.preprocess(image, for_recognition=True)
