"""
Timing Instrumentation for Face Recognition Pipeline

Tracks and logs timing for:
i)   Capture time
ii)  Feature extraction (ROI + preprocessing + QCNN)
iii) Face classification time
iv)  Recognition time
v)   PQC operations (encryption/decryption/retrieval)
vi)  Total pipeline time
"""

import time
import os
import csv
from datetime import datetime
from typing import Dict, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingData:
    """Data class to store timing measurements"""
    capture: float = 0.0
    roi_detection: float = 0.0
    preprocessing: float = 0.0
    qcnn_forward: float = 0.0
    classification: float = 0.0
    recognition: float = 0.0
    db_query: float = 0.0
    similarity: float = 0.0
    decryption: float = 0.0
    encryption: float = 0.0
    retrieval: float = 0.0
    total: float = 0.0

    # Metadata
    operation: str = ""
    user: str = ""
    result: str = ""
    timestamp: str = ""

    def feature_extraction_total(self) -> float:
        """Total feature extraction time"""
        return self.roi_detection + self.preprocessing + self.qcnn_forward

    def pqc_total(self) -> float:
        """Total PQC operations time"""
        return self.decryption + self.encryption + self.retrieval


class Timer:
    """Context manager for timing code blocks"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        if self.elapsed > 0:
            return self.elapsed
        return time.perf_counter() - self.start_time


class TimingLogger:
    """
    Logger for timing data

    Logs timing measurements to CSV file and provides console output.
    """

    def __init__(self, log_dir: str = None):
        """
        Initialize timing logger

        Args:
            log_dir: Directory for log files (default: quantum_face_recognition/logs)
        """
        if log_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.log_dir = os.path.join(base_dir, "logs")
        else:
            self.log_dir = log_dir

        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "timing_log.csv")

        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            self._write_headers()

        # Current timing session
        self.current_timing = TimingData()
        self.session_start = None

    def _write_headers(self):
        """Write CSV headers"""
        headers = [
            'timestamp', 'operation', 'capture_sec', 'roi_sec', 'preprocess_sec',
            'qcnn_sec', 'classification_sec', 'recognition_sec', 'db_query_sec',
            'similarity_sec', 'decrypt_sec', 'encrypt_sec', 'retrieval_sec',
            'total_sec', 'user', 'result'
        ]
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def start_session(self, operation: str = ""):
        """Start a new timing session"""
        self.current_timing = TimingData()
        self.current_timing.operation = operation
        self.current_timing.timestamp = datetime.now().isoformat()
        self.session_start = time.perf_counter()

    def end_session(self, user: str = "", result: str = ""):
        """End timing session and calculate total"""
        if self.session_start:
            self.current_timing.total = time.perf_counter() - self.session_start
        self.current_timing.user = user
        self.current_timing.result = result

    @contextmanager
    def time_capture(self):
        """Time face capture"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.capture = timer.elapsed

    @contextmanager
    def time_roi_detection(self):
        """Time ROI/face detection"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.roi_detection = timer.elapsed

    @contextmanager
    def time_preprocessing(self):
        """Time preprocessing"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.preprocessing = timer.elapsed

    @contextmanager
    def time_qcnn_forward(self):
        """Time QCNN forward pass"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.qcnn_forward = timer.elapsed

    @contextmanager
    def time_classification(self):
        """Time face/not-face classification"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.classification = timer.elapsed

    @contextmanager
    def time_recognition(self):
        """Time recognition/matching"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.recognition = timer.elapsed

    @contextmanager
    def time_db_query(self):
        """Time database query"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.db_query = timer.elapsed

    @contextmanager
    def time_similarity(self):
        """Time similarity computation"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.similarity = timer.elapsed

    @contextmanager
    def time_decryption(self):
        """Time PQC decryption"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.decryption = timer.elapsed

    @contextmanager
    def time_encryption(self):
        """Time PQC encryption"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.encryption = timer.elapsed

    @contextmanager
    def time_retrieval(self):
        """Time data retrieval"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.retrieval = timer.elapsed

    # Accumulating versions for multiple operations (e.g., loading multiple embeddings)
    @contextmanager
    def time_decryption_accumulate(self):
        """Time PQC decryption (accumulates across multiple calls)"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.decryption += timer.elapsed  # Accumulate, not overwrite

    @contextmanager
    def time_retrieval_accumulate(self):
        """Time data retrieval (accumulates across multiple calls)"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.retrieval += timer.elapsed  # Accumulate, not overwrite

    @contextmanager
    def time_similarity_accumulate(self):
        """Time similarity computation (accumulates across multiple calls)"""
        timer = Timer()
        with timer:
            yield
        self.current_timing.similarity += timer.elapsed  # Accumulate, not overwrite

    def log_to_file(self):
        """Log current timing data to CSV file"""
        t = self.current_timing
        row = [
            t.timestamp, t.operation, f"{t.capture:.4f}", f"{t.roi_detection:.4f}",
            f"{t.preprocessing:.4f}", f"{t.qcnn_forward:.4f}", f"{t.classification:.4f}",
            f"{t.recognition:.4f}", f"{t.db_query:.4f}", f"{t.similarity:.4f}",
            f"{t.decryption:.4f}", f"{t.encryption:.4f}", f"{t.retrieval:.4f}",
            f"{t.total:.4f}", t.user, t.result
        ]
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def print_report(self):
        """Print timing report to console"""
        t = self.current_timing

        print("\n" + "=" * 60)
        print("AUTHENTICATION TIMING REPORT")
        print("=" * 60)
        print(f"  Operation: {t.operation}")
        print(f"  Timestamp: {t.timestamp}")
        print("-" * 60)
        print(f"  i)   Capture:              {t.capture:.3f} sec")
        print(f"  ii)  Feature Extraction:   {t.feature_extraction_total():.3f} sec")
        print(f"       - ROI Detection:      {t.roi_detection:.3f} sec")
        print(f"       - Preprocessing:      {t.preprocessing:.3f} sec")
        print(f"       - QCNN Forward:       {t.qcnn_forward:.3f} sec")
        print(f"  iii) Face Classification:  {t.classification:.3f} sec")
        print(f"  iv)  Recognition:          {t.recognition:.3f} sec")
        if t.db_query > 0 or t.similarity > 0:
            print(f"       - DB Query:           {t.db_query:.3f} sec")
            print(f"       - Similarity:         {t.similarity:.3f} sec")
        print(f"  v)   PQC Operations:       {t.pqc_total():.3f} sec")
        if t.pqc_total() > 0:
            print(f"       - Decryption:         {t.decryption:.3f} sec")
            print(f"       - Encryption:         {t.encryption:.3f} sec")
            print(f"       - Retrieval:          {t.retrieval:.3f} sec")
        print("-" * 60)
        print(f"  vi)  TOTAL PIPELINE:       {t.total:.3f} sec")
        print("=" * 60)
        if t.user:
            print(f"  User: {t.user}")
        if t.result:
            print(f"  Result: {t.result}")
        print("=" * 60)

    def get_timing_data(self) -> TimingData:
        """Get current timing data"""
        return self.current_timing

    def get_timing_dict(self) -> Dict:
        """Get timing data as dictionary"""
        t = self.current_timing
        return {
            'timestamp': t.timestamp,
            'operation': t.operation,
            'capture': t.capture,
            'roi_detection': t.roi_detection,
            'preprocessing': t.preprocessing,
            'qcnn_forward': t.qcnn_forward,
            'feature_extraction_total': t.feature_extraction_total(),
            'classification': t.classification,
            'recognition': t.recognition,
            'db_query': t.db_query,
            'similarity': t.similarity,
            'decryption': t.decryption,
            'encryption': t.encryption,
            'retrieval': t.retrieval,
            'pqc_total': t.pqc_total(),
            'total': t.total,
            'user': t.user,
            'result': t.result
        }


# Global timing logger instance
_global_logger: Optional[TimingLogger] = None


def get_timing_logger() -> TimingLogger:
    """Get or create global timing logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TimingLogger()
    return _global_logger


def reset_timing_logger():
    """Reset global timing logger"""
    global _global_logger
    _global_logger = None


if __name__ == "__main__":
    # Test timing instrumentation
    print("Testing Timing Instrumentation...")
    print("=" * 60)

    logger = TimingLogger()
    logger.start_session("LOGIN_TEST")

    # Simulate operations with timing
    with logger.time_capture():
        time.sleep(0.1)  # Simulate capture

    with logger.time_roi_detection():
        time.sleep(0.05)  # Simulate ROI detection

    with logger.time_preprocessing():
        time.sleep(0.02)  # Simulate preprocessing

    with logger.time_qcnn_forward():
        time.sleep(0.08)  # Simulate QCNN

    with logger.time_classification():
        time.sleep(0.01)  # Simulate classification

    with logger.time_recognition():
        time.sleep(0.05)  # Simulate recognition

    with logger.time_decryption():
        time.sleep(0.15)  # Simulate decryption

    logger.end_session(user="Test User", result="SUCCESS")

    # Print report
    logger.print_report()

    # Log to file
    logger.log_to_file()
    print(f"\nTiming logged to: {logger.log_file}")
