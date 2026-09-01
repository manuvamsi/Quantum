"""
Haar Cascade face detection + mask detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Detect faces and filter out masks
    """
    
    def __init__(self):
        # Load Haar cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces in image
        
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    
    def is_wearing_mask(self, face_roi: np.ndarray) -> bool:
        """
        Check if face is wearing mask
        
        Method: Check variance in lower 40% of face
        Masked faces have low variance (uniform color)
        """
        h, w = face_roi.shape[:2]
        lower_face = face_roi[int(h*0.6):, :]
        
        # Convert to grayscale and compute variance
        if len(lower_face.shape) == 3:
            lower_face = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        
        variance = np.var(lower_face)
        
        # Threshold: masked faces have variance < 200
        return variance < 200
    
    def detect_and_filter(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces and filter out masked ones
        """
        faces = self.detect_faces(image)
        valid_faces = []
        
        for (x, y, w, h) in faces:
            roi = image[y:y+h, x:x+w]
            if not self.is_wearing_mask(roi):
                valid_faces.append((x, y, w, h))
        
        return valid_faces
