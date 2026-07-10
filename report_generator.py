from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict


def generate_report(report_dir: str, payload: Dict[str, object]) -> Path:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"defect_report_{timestamp}.txt"

    lines = [
        "Quantum Vision Defect Detection Report",
        "=" * 40,
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Final Decision: {payload['final_label']}",
        f"Classical Defect Type: {payload['defect_type']}",
        f"Classical Score: {payload['classical_score']:.3f}",
        f"Quantum Confidence: {payload['quantum_confidence']:.3f}",
        "",
        "Metrics:",
    ]

    metrics = payload.get("metrics", {})
    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.4f}")

    lines.extend(
        [
            "",
            "Notes:",
            "- Classical stage estimates surface irregularities from texture and edges.",
            "- Quantum stage evaluates compressed features through a small variational-style circuit.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
