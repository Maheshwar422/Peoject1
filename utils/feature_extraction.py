"""Image feature extraction for quantum vision defect detection."""

import cv2
import numpy as np


def extract_features(image: np.ndarray, n_patches: int = 4) -> dict:
    """
    Extract statistical and texture features from an image for quantum encoding.

    Returns a dict with normalized feature vector and diagnostic metadata.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Global statistics
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))

    # Edge density via Canny
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.sum(edges > 0) / edges.size)

    # Local variance (texture irregularity indicator)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = float(np.var(laplacian))

    # Patch-level anomaly scores
    patch_h, patch_w = h // n_patches, w // n_patches
    patch_scores = []
    for i in range(n_patches):
        for j in range(n_patches):
            patch = gray[i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
            patch_edges = cv2.Canny(patch, 80, 180)
            patch_scores.append(float(np.sum(patch_edges > 0) / patch_edges.size))

    patch_anomaly = float(np.std(patch_scores))
    max_patch_edge = float(np.max(patch_scores))

    # Contour irregularity
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    contour_complexity = 0.0
    if contours:
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50]
        if areas:
            contour_complexity = float(np.std(areas) / (np.mean(areas) + 1e-6))

    # Normalize features to [0, 1] for quantum angle encoding
    feature_vector = np.array(
        [
            mean_intensity / 255.0,
            min(std_intensity / 80.0, 1.0),
            min(edge_density / 0.15, 1.0),
            min(texture_variance / 5000.0, 1.0),
            min(patch_anomaly / 0.08, 1.0),
            min(max_patch_edge / 0.2, 1.0),
            min(contour_complexity / 2.0, 1.0),
            min(contour_count / 50.0, 1.0),
        ],
        dtype=np.float64,
    )

    return {
        "feature_vector": feature_vector,
        "edges": edges,
        "gray": gray,
        "patch_scores": patch_scores,
        "metrics": {
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "edge_density": round(edge_density, 4),
            "texture_variance": round(texture_variance, 2),
            "patch_anomaly": round(patch_anomaly, 4),
            "max_patch_edge": round(max_patch_edge, 4),
            "contour_count": contour_count,
            "contour_complexity": round(contour_complexity, 4),
        },
    }


def create_defect_heatmap(image: np.ndarray, n_patches: int = 8) -> np.ndarray:
    """Generate a heatmap highlighting regions with high edge density."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    patch_h, patch_w = max(h // n_patches, 1), max(w // n_patches, 1)
    for i in range(n_patches):
        for j in range(n_patches):
            y1, y2 = i * patch_h, min((i + 1) * patch_h, h)
            x1, x2 = j * patch_w, min((j + 1) * patch_w, w)
            patch = gray[y1:y2, x1:x2]
            edges = cv2.Canny(patch, 80, 180)
            score = np.sum(edges > 0) / edges.size
            heatmap[y1:y2, x1:x2] = score

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)
    return overlay
