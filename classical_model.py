from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class ClassicalAnalysisResult:
    features: np.ndarray
    defect_score: float
    defect_type: str
    heatmap: np.ndarray
    metrics: Dict[str, float]


class ClassicalDefectAnalyzer:
    """Lightweight classical vision stage for hackathon demos.

    The analyzer uses edge density, texture variance, and local contrast to
    approximate whether a product image contains visible defects.
    """

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (256, 256))
        return resized

    def analyze(self, image_bgr: np.ndarray) -> ClassicalAnalysisResult:
        image = self.preprocess(image_bgr)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 60, 160)
        laplacian = cv2.Laplacian(blur, cv2.CV_32F)
        local_diff = cv2.absdiff(gray, cv2.medianBlur(gray, 9))

        edge_density = float(np.count_nonzero(edges)) / edges.size
        texture_variance = float(np.var(laplacian)) / 1000.0
        contrast_score = float(np.mean(local_diff)) / 255.0
        brightness = float(np.mean(gray)) / 255.0

        heatmap_raw = cv2.normalize(
            edges.astype(np.float32) + local_diff.astype(np.float32),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

        feature_vector = np.array(
            [edge_density, texture_variance, contrast_score, brightness],
            dtype=np.float32,
        )

        defect_score = float(
            np.clip(
                0.45 * edge_density
                + 0.35 * min(texture_variance, 1.0)
                + 0.20 * contrast_score,
                0.0,
                1.0,
            )
        )

        if defect_score > 0.60:
            defect_type = "Crack / severe surface anomaly"
        elif defect_score > 0.38:
            defect_type = "Scratch / moderate texture anomaly"
        else:
            defect_type = "No obvious defect"

        return ClassicalAnalysisResult(
            features=feature_vector,
            defect_score=defect_score,
            defect_type=defect_type,
            heatmap=heatmap_color,
            metrics={
                "edge_density": edge_density,
                "texture_variance": texture_variance,
                "contrast_score": contrast_score,
                "brightness": brightness,
            },
        )
