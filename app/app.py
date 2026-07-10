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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .metric-container {
        display: flex;
        gap: 1.2rem;
        margin-bottom: 2.5rem;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        border-left: 6px solid #00f2fe;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(255, 255, 255, 0.75);
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.5rem;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    
    .verdict-banner-pass {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 16px;
        padding: 1.8rem;
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(56, 239, 125, 0.25);
    }
    .verdict-banner-fail {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        color: white;
        border-radius: 16px;
        padding: 1.8rem;
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(239, 71, 58, 0.25);
    }
    .verdict-icon {
        font-size: 2.8rem;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        width: 65px;
        height: 65px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .verdict-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .verdict-subtitle {
        font-size: 1.1rem;
        margin-top: 0.3rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.3);
    }
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
st.markdown(
    """
    <div class="header-container">
        <h1 style="color: #ffffff; font-weight: 800; font-size: 2.6rem; margin: 0; display: flex; align-items: center; gap: 0.8rem; letter-spacing: -0.5px;">
            <span>🔬</span> QT-2.3 Quantum Vision System
        </h1>
        <p style="color: #00f2fe; font-size: 1.05rem; font-weight: 600; margin: 0.5rem 0 0 0; text-transform: uppercase; letter-spacing: 2px;">
            Automated Manufacturing Quality Inspection &nbsp;|&nbsp; Quant-A-thon 2026
        </p>
    </div>
    """,
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
banner_class = "verdict-banner-fail" if final["status"] == "DEFECTIVE" else "verdict-banner-pass"
verdict_icon = "❌" if final["status"] == "DEFECTIVE" else "✅"
st.markdown(
    f"""
    <div class="{banner_class}">
        <div class="verdict-icon">{verdict_icon}</div>
        <div>
            <div class="verdict-title">Final Verdict: {final["status"]} &nbsp;|&nbsp; Confidence: {final["confidence"]:.1%}</div>
            <div class="verdict-subtitle"><strong>Recommendation:</strong> {final['recommendation']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Metrics row ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="metric-container">
        <div class="metric-card" style="border-left-color: #00f2fe;">
            <div class="metric-label">Fused Score</div>
            <div class="metric-value">{final['fused_score']:.3f}</div>
        </div>
        <div class="metric-card" style="border-left-color: #ff007f;">
            <div class="metric-label">Classical Score</div>
            <div class="metric-value">{classical['defect_score']:.3f}</div>
        </div>
        <div class="metric-card" style="border-left-color: #7f00ff;">
            <div class="metric-label">Quantum Probability</div>
            <div class="metric-value">{quantum['defect_probability']:.3f}</div>
        </div>
        <div class="metric-card" style="border-left-color: #39ff14;">
            <div class="metric-label">Models Agree</div>
            <div class="metric-value">{"Yes ✅" if final["models_agree"] else "No ⚠️"}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
