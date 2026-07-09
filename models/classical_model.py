"""Classical computer-vision defect detection baseline."""

import cv2
import numpy as np

from utils.feature_extraction import extract_features


def _classify_defect_type(metrics: dict) -> str:
    edge = metrics.get("edge_density", 0)
    texture = metrics.get("texture_variance", 0)
    patch = metrics.get("patch_anomaly", 0)

    if edge > 0.12 and patch > 0.06:
        return "Surface Scratch / Crack"
    if texture > 3000 and edge > 0.08:
        return "Texture Anomaly"
    if patch > 0.05:
        return "Localized Defect"
    if edge > 0.1:
        return "Edge Irregularity"
    return "General Defect"


def classical_predict(image: np.ndarray) -> dict:
    """
    Detect defects using edge density, texture variance, and patch anomaly scoring.
    Returns structured result with verdict, confidence, and visualization data.
    """
    data = extract_features(image)
    metrics = data["metrics"]
    edges = data["edges"]

    # Weighted defect score (heuristic, tuned for manufacturing inspection)
    score = (
        0.30 * min(metrics["edge_density"] / 0.12, 1.0)
        + 0.25 * min(metrics["texture_variance"] / 4000, 1.0)
        + 0.25 * min(metrics["patch_anomaly"] / 0.07, 1.0)
        + 0.10 * min(metrics["contour_complexity"] / 1.5, 1.0)
        + 0.10 * min(metrics["max_patch_edge"] / 0.15, 1.0)
    )

    is_defect = score > 0.12
    confidence = score if is_defect else 1.0 - score

    reasoning = []
    if metrics["edge_density"] > 0.08:
        reasoning.append(f"Edge density ({metrics['edge_density']:.4f}) exceeds threshold")
    if metrics["texture_variance"] > 2000:
        reasoning.append(f"Texture variance ({metrics['texture_variance']:.0f}) indicates surface irregularity")
    if metrics["patch_anomaly"] > 0.04:
        reasoning.append(f"Patch anomaly ({metrics['patch_anomaly']:.4f}) shows non-uniform regions")
    if not reasoning:
        reasoning.append("All surface metrics within acceptable manufacturing tolerances")

    return {
        "verdict": "DEFECTIVE" if is_defect else "GOOD",
        "confidence": round(min(max(confidence, 0.0), 1.0), 4),
        "defect_score": round(score, 4),
        "defect_type": _classify_defect_type(metrics) if is_defect else "None",
        "reasoning": reasoning,
        "edges": edges,
        "features": data,
    }
