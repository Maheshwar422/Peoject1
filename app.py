from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from models.classical_model import ClassicalDefectAnalyzer
from models.quantum_model import QuantumDefectClassifier
from reports.report_generator import generate_report


st.set_page_config(page_title="QT-2.3 Defect Detection", layout="wide")

st.title("QT-2.3 Quantum Vision Defect Detection")
st.write(
    "Upload a manufacturing product image to run a hybrid classical + quantum "
    "defect analysis demo."
)

uploaded_file = st.file_uploader(
    "Choose a product image", type=["png", "jpg", "jpeg", "bmp"]
)

analyzer = ClassicalDefectAnalyzer()
quantum_classifier = QuantumDefectClassifier()

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not decode the uploaded image.")
    else:
        classical = analyzer.analyze(image_bgr)
        quantum = quantum_classifier.predict(classical.features)

        final_label = (
            "Defective"
            if classical.defect_score >= 0.40 or quantum.label == "Defective"
            else "Likely OK"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Defect Heatmap")
            st.image(
                cv2.cvtColor(classical.heatmap, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        st.subheader("Decision Summary")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Final Decision", final_label)
        metric2.metric("Classical Score", f"{classical.defect_score:.3f}")
        metric3.metric("Quantum Confidence", f"{quantum.confidence:.3f}")

        st.write(f"**Classical defect type:** {classical.defect_type}")
        st.write(f"**Quantum label:** {quantum.label}")

        st.subheader("Feature Metrics")
        st.json(classical.metrics)

        report_payload = {
            "final_label": final_label,
            "defect_type": classical.defect_type,
            "classical_score": classical.defect_score,
            "quantum_confidence": quantum.confidence,
            "metrics": classical.metrics,
        }

        if st.button("Generate defect report"):
            report_path = generate_report("reports", report_payload)
            st.success(f"Report saved to {Path(report_path)}")

        st.subheader("Quantum State Probabilities")
        st.bar_chart(quantum.state_probabilities)
else:
    st.info("Upload an image to start the inspection demo.")
