"""Defect detection report generation."""

import json
from datetime import datetime
from typing import Any


def generate_report(
    filename: str,
    classical_result: dict,
    quantum_result: dict,
    final_verdict: dict,
    features: dict,
) -> dict:
    """Build a structured defect detection report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = {
        "report_title": "Quantum Vision Defect Detection Report",
        "task_id": "QT-2.3",
        "generated_at": timestamp,
        "input": {
            "filename": filename,
            "image_metrics": features.get("metrics", {}),
        },
        "classical_analysis": {
            "verdict": classical_result["verdict"],
            "confidence": classical_result["confidence"],
            "defect_type": classical_result.get("defect_type", "N/A"),
            "reasoning": classical_result.get("reasoning", []),
        },
        "quantum_analysis": {
            "verdict": quantum_result["verdict"],
            "confidence": quantum_result["confidence"],
            "quantum_score": quantum_result.get("quantum_score", 0.0),
            "circuit_depth": quantum_result.get("circuit_depth", 0),
            "reasoning": quantum_result.get("reasoning", []),
        },
        "final_verdict": {
            "status": final_verdict["status"],
            "confidence": final_verdict["confidence"],
            "recommendation": final_verdict["recommendation"],
        },
        "quality_assessment": _quality_summary(final_verdict, features),
    }
    return report


def _quality_summary(final_verdict: dict, features: dict) -> dict:
    metrics = features.get("metrics", {})
    if final_verdict["status"] == "DEFECTIVE":
        return {
            "inspection_result": "FAIL",
            "action_required": "Remove from production line for manual review.",
            "risk_level": "HIGH" if final_verdict["confidence"] > 0.75 else "MEDIUM",
            "primary_indicators": _top_indicators(metrics),
        }
    return {
        "inspection_result": "PASS",
        "action_required": "Proceed to next stage.",
        "risk_level": "LOW",
        "primary_indicators": _top_indicators(metrics),
    }


def _top_indicators(metrics: dict) -> list[str]:
    indicators = []
    if metrics.get("edge_density", 0) > 0.08:
        indicators.append("Elevated edge density detected")
    if metrics.get("patch_anomaly", 0) > 0.04:
        indicators.append("Non-uniform surface texture across patches")
    if metrics.get("texture_variance", 0) > 2000:
        indicators.append("High local texture variance")
    if metrics.get("contour_complexity", 0) > 1.0:
        indicators.append("Irregular contour patterns")
    if not indicators:
        indicators.append("Surface metrics within normal range")
    return indicators


def report_to_text(report: dict) -> str:
    """Render report as human-readable text."""
    lines = [
        "=" * 60,
        report["report_title"],
        f"Task: {report['task_id']} | Generated: {report['generated_at']}",
        "=" * 60,
        "",
        "INPUT",
        f"  File: {report['input']['filename']}",
        "",
        "IMAGE METRICS",
    ]
    for key, value in report["input"]["image_metrics"].items():
        lines.append(f"  {key}: {value}")

    lines.extend(
        [
            "",
            "CLASSICAL ANALYSIS",
            f"  Verdict: {report['classical_analysis']['verdict']}",
            f"  Confidence: {report['classical_analysis']['confidence']:.1%}",
            f"  Defect Type: {report['classical_analysis']['defect_type']}",
        ]
    )
    for reason in report["classical_analysis"]["reasoning"]:
        lines.append(f"    - {reason}")

    lines.extend(
        [
            "",
            "QUANTUM VISION ANALYSIS",
            f"  Verdict: {report['quantum_analysis']['verdict']}",
            f"  Confidence: {report['quantum_analysis']['confidence']:.1%}",
            f"  Quantum Score: {report['quantum_analysis']['quantum_score']:.4f}",
            f"  Circuit Depth: {report['quantum_analysis']['circuit_depth']}",
        ]
    )
    for reason in report["quantum_analysis"]["reasoning"]:
        lines.append(f"    - {reason}")

    fv = report["final_verdict"]
    qa = report["quality_assessment"]
    lines.extend(
        [
            "",
            "FINAL VERDICT",
            f"  Status: {fv['status']}",
            f"  Confidence: {fv['confidence']:.1%}",
            f"  Recommendation: {fv['recommendation']}",
            "",
            "QUALITY ASSESSMENT",
            f"  Inspection Result: {qa['inspection_result']}",
            f"  Risk Level: {qa['risk_level']}",
            f"  Action: {qa['action_required']}",
            "  Indicators:",
        ]
    )
    for ind in qa["primary_indicators"]:
        lines.append(f"    - {ind}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def report_to_json(report: dict) -> str:
    """Serialize report to JSON."""
    return json.dumps(report, indent=2)
