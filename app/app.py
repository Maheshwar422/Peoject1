"""
QT-2.3 Quantum Vision Defect Detection System
Quant-A-thon 2026 — Manufacturing Quality Inspection
"""

import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st

from models.hybrid_model import hybrid_predict
from models.quantum_model import get_circuit_info
from utils.feature_extraction import create_defect_heatmap
from utils.report_generator import generate_report, report_to_json, report_to_text

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QT-2.3 Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .sub-header  { color: #4a4a6a; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-radius: 12px; padding: 1rem 1.2rem;
        border-left: 4px solid #667eea;
    }
    .pass  { color: #27ae60; font-weight: 700; font-size: 1.4rem; }
    .fail  { color: #e74c3c; font-weight: 700; font-size: 1.4rem; }
    .stDownloadButton button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=80)
    st.title("QT-2.3")
    st.markdown("**Quantum Vision Defect Detection**")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Hybrid **quantum-classical** vision system for automated "
        "manufacturing quality inspection."
    )
    st.markdown("---")
    st.markdown("### Quantum Circuit")
    info = get_circuit_info()
    for k, v in info.items():
        st.markdown(f"**{k.replace('_', ' ').title()}:** `{v}`")
    st.markdown("---")
    st.markdown("### Evaluation Focus")
    st.progress(0.35, text="Detection Accuracy — 35%")
    st.progress(0.20, text="Innovation (Quantum) — 20%")
    st.progress(0.20, text="Efficiency — 20%")
    st.progress(0.15, text="Usability — 15%")
    st.progress(0.10, text="Presentation — 10%")

# ── Main header ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔬 Quantum Vision Defect Detection System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Quant-A-thon 2026 &nbsp;|&nbsp; QT-2.3 &nbsp;|&nbsp; '
    "Automated Manufacturing Quality Inspection</p>",
    unsafe_allow_html=True,
)

# ── Upload ─────────────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([2, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Upload product image for inspection",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a manufacturing product image to analyze for defects.",
    )
with col_info:
    st.info("Supported: JPG, PNG, BMP  \nMax recommended: 4096×4096 px")

if uploaded is None:
    st.markdown("---")
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1️⃣ Feature Extraction")
        st.markdown(
            "Extracts edge density, texture variance, patch anomaly scores "
            "and contour metrics from the uploaded image."
        )
    with c2:
        st.markdown("#### 2️⃣ Quantum Vision")
        st.markdown(
            "Encodes features into a 4-qubit variational quantum circuit "
            "using angle encoding and ring entanglement."
        )
    with c3:
        st.markdown("#### 3️⃣ Hybrid Fusion")
        st.markdown(
            "Fuses classical CV and quantum predictions (55/45 weighting) "
            "to produce a final verdict with confidence score."
        )
    st.stop()

# ── Decode image ───────────────────────────────────────────────────────────────
file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image is None:
    st.error("Could not decode the uploaded image. Please try a different file.")
    st.stop()

# ── Run analysis ───────────────────────────────────────────────────────────────
with st.spinner("Running quantum-classical defect analysis…"):
    result = hybrid_predict(image)
    heatmap = create_defect_heatmap(image)

classical = result["classical"]
quantum = result["quantum"]
final = result["final"]
features = result["features"]

report = generate_report(
    filename=uploaded.name,
    classical_result=classical,
    quantum_result=quantum,
    final_verdict=final,
    features=features,
)

# ── Final verdict banner ───────────────────────────────────────────────────────
st.markdown("---")
verdict_class = "fail" if final["status"] == "DEFECTIVE" else "pass"
verdict_icon = "❌" if final["status"] == "DEFECTIVE" else "✅"
st.markdown(
    f'<p class="{verdict_class}">{verdict_icon} FINAL VERDICT: {final["status"]} '
    f'&nbsp;|&nbsp; Confidence: {final["confidence"]:.1%}</p>',
    unsafe_allow_html=True,
)
st.markdown(f"**Recommendation:** {final['recommendation']}")

# ── Metrics row ────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Fused Score", f"{final['fused_score']:.3f}")
m2.metric("Classical Score", f"{classical['defect_score']:.3f}")
m3.metric("Quantum Probability", f"{quantum['defect_probability']:.3f}")
m4.metric("Models Agree", "Yes ✅" if final["models_agree"] else "No ⚠️")

# ── Image panels ───────────────────────────────────────────────────────────────
st.markdown("### Visual Analysis")
img_col1, img_col2, img_col3 = st.columns(3)
with img_col1:
    st.markdown("**Original Image**")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
with img_col2:
    st.markdown("**Edge Map (Classical)**")
    st.image(classical["edges"], use_container_width=True, clamp=True)
with img_col3:
    st.markdown("**Defect Heatmap**")
    st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)

# ── Model details ──────────────────────────────────────────────────────────────
st.markdown("### Model Analysis")
tab_classical, tab_quantum, tab_metrics, tab_report = st.tabs(
    ["Classical CV", "Quantum Vision", "Image Metrics", "Detection Report"]
)

with tab_classical:
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"**Verdict:** `{classical['verdict']}`")
        st.markdown(f"**Confidence:** `{classical['confidence']:.1%}`")
        st.markdown(f"**Defect Type:** `{classical['defect_type']}`")
    with cc2:
        st.markdown("**Reasoning:**")
        for r in classical["reasoning"]:
            st.markdown(f"- {r}")

with tab_quantum:
    qc1, qc2 = st.columns(2)
    with qc1:
        st.markdown(f"**Verdict:** `{quantum['verdict']}`")
        st.markdown(f"**Confidence:** `{quantum['confidence']:.1%}`")
        st.markdown(f"**Quantum Score:** `{quantum['quantum_score']}`")
        st.markdown(f"**Circuit Depth:** `{quantum['circuit_depth']}`")
    with qc2:
        st.markdown("**Reasoning:**")
        for r in quantum["reasoning"]:
            st.markdown(f"- {r}")
    st.markdown("**Quantum Circuit Architecture:**")
    st.code(
        """
        Feature Encoding (Angle):
          RY(f[i*2] * π) + RZ(f[i*2+1] * π)  on each of 4 qubits

        Variational Layers (×3):
          RY(weights) on each qubit
          CNOT ring: 0→1→2→3→0

        Measurement:
          ⟨PauliZ₀⟩  →  defect probability via sigmoid
        """,
        language="text",
    )

with tab_metrics:
    metrics = features["metrics"]
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**Surface Metrics**")
        for k, v in metrics.items():
            st.markdown(f"- **{k.replace('_', ' ').title()}:** `{v}`")
    with mc2:
        st.markdown("**Patch Edge Scores**")
        patch_scores = features["patch_scores"]
        st.bar_chart({"edge_density": patch_scores})

with tab_report:
    st.markdown("#### Defect Detection Report")
    report_text = report_to_text(report)
    st.text(report_text)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report_text,
            file_name=f"defect_report_{uploaded.name.rsplit('.', 1)[0]}.txt",
            mime="text/plain",
        )
    with dl2:
        st.download_button(
            label="📋 Download Report (JSON)",
            data=report_to_json(report),
            file_name=f"defect_report_{uploaded.name.rsplit('.', 1)[0]}.json",
            mime="application/json",
        )

st.markdown("---")
st.caption("QT-2.3 Quantum Vision Defect Detection | Quant-A-thon 2026 | PennyLane + OpenCV + Streamlit")
