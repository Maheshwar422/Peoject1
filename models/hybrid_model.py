"""Hybrid quantum-classical defect detection orchestrator."""

from models.classical_model import classical_predict
from models.quantum_model import quantum_predict


def hybrid_predict(image) -> dict:
    """
    Run both classical and quantum vision models, then fuse predictions.
    Quantum weight: 0.55 (innovation emphasis), Classical: 0.45 (accuracy baseline).
    """
    classical = classical_predict(image)
    quantum = quantum_predict(classical["features"]["feature_vector"])

    q_weight, c_weight = 0.55, 0.45

    c_score = classical["defect_score"]
    q_score = quantum["defect_probability"]
    fused_score = q_weight * q_score + c_weight * c_score

    agree = classical["verdict"] == quantum["verdict"]
    if agree:
        is_defect = classical["verdict"] == "DEFECTIVE"
        confidence = max(q_score, c_score) if is_defect else min(1 - q_score, 1 - c_score)
    else:
        is_defect = fused_score > 0.45
        confidence = fused_score if is_defect else 1.0 - fused_score

    confidence = min(confidence * 1.1, 1.0)

    if is_defect:
        recommendation = (
            f"Reject product. Primary concern: {classical['defect_type']}. "
            "Schedule manual inspection."
        )
    else:
        recommendation = "Product passes quality inspection. Approve for shipment."

    return {
        "classical": classical,
        "quantum": quantum,
        "final": {
            "status": "DEFECTIVE" if is_defect else "GOOD",
            "confidence": round(min(max(confidence, 0.0), 1.0), 4),
            "fused_score": round(fused_score, 4),
            "models_agree": agree,
            "recommendation": recommendation,
        },
        "features": classical["features"],
    }
