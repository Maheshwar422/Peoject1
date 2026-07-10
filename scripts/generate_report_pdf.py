"""Generate QT-2.3 hackathon submission PDF report."""

import os
from datetime import datetime

from fpdf import FPDF

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "QT-2.3_Defect_Detection_Report.pdf")


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(80, 80, 120)
        self.cell(0, 8, "QT-2.3 Quantum Vision Defect Detection | Quant-A-thon 2026", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 60)
        self.cell(0, 10, title)
        self.ln(8)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def bullet(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.set_x(x + 8)
        self.multi_cell(0, 6, f"- {text}")
        self.set_x(x)


def build_pdf():
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title page content
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 15, "Quantum Vision Defect Detection System", align="C")
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(74, 74, 106)
    pdf.cell(0, 10, "Quant-A-thon 2026  |  Task QT-2.3", align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, f"Submission Report - {datetime.now().strftime('%B %d, %Y')}", align="C")
    pdf.ln(20)

    pdf.section_title("1. Problem Statement")
    pdf.body_text(
        "Manufacturing quality inspection requires fast, accurate detection of product defects. "
        "Manual inspection is slow and inconsistent. This project delivers an automated "
        "Quantum Vision-based Defect Detection System that analyzes product images and "
        "generates inspection reports with pass/fail verdicts."
    )

    pdf.section_title("2. Objectives")
    pdf.bullet("Detect product defects in manufacturing images")
    pdf.bullet("Reduce inspection time through automated analysis")
    pdf.bullet("Improve quality assurance with confidence-scored reports")
    pdf.ln(5)

    pdf.section_title("3. Solution Overview")
    pdf.body_text(
        "We built a hybrid quantum-classical vision pipeline. Classical computer vision "
        "extracts edge density, texture variance, patch anomaly scores, and contour metrics. "
        "These features are angle-encoded into a 4-qubit variational quantum circuit (PennyLane). "
        "Classical and quantum predictions are fused to produce a final verdict with confidence."
    )

    pdf.section_title("4. Deliverables")
    pdf.bullet("Image Analysis Model: Classical CV + 4-qubit VQC (PennyLane)")
    pdf.bullet("Defect Detection Report: Downloadable TXT and JSON reports from the UI")
    pdf.bullet("User Interface: Streamlit web application with visual heatmaps and metrics")
    pdf.ln(5)

    pdf.section_title("5. Architecture")
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(50, 50, 50)
    architecture = (
        "Upload Image -> Feature Extraction (OpenCV)\n"
        "  -> Classical Model (weighted scoring)\n"
        "  -> Quantum Vision Model (4-qubit VQC, angle encoding)\n"
        "  -> Hybrid Fusion (55% quantum / 45% classical)\n"
        "  -> Final Verdict + Detection Report"
    )
    pdf.multi_cell(0, 5, architecture)
    pdf.ln(5)

    pdf.section_title("6. Quantum Innovation")
    pdf.bullet("Angle encoding of 8 image features onto 4 qubits (RY + RZ rotations)")
    pdf.bullet("3-layer variational circuit with ring CNOT entanglement topology")
    pdf.bullet("PauliZ measurement mapped to defect probability via sigmoid activation")
    pdf.bullet("Captures non-linear feature correlations beyond classical linear models")
    pdf.ln(5)

    pdf.section_title("7. Tech Stack")
    pdf.bullet("OpenCV - image processing and feature extraction")
    pdf.bullet("PennyLane - quantum machine learning framework")
    pdf.bullet("Streamlit - interactive web interface")
    pdf.bullet("NumPy - numerical computation")
    pdf.ln(5)

    pdf.section_title("8. How to Run")
    pdf.set_font("Courier", "", 10)
    pdf.multi_cell(
        0, 6,
        "pip install -r requirements.txt\n"
        "streamlit run app/app.py\n"
        "Open http://localhost:8501 and upload a product image"
    )
    pdf.ln(5)

    pdf.section_title("9. Test Results")
    pdf.body_text("Validated on three sample images:")
    pdf.bullet("good_product.png -> GOOD (89% confidence)")
    pdf.bullet("defect_scratch.png -> DEFECTIVE (69% confidence)")
    pdf.bullet("defect_texture.png -> DEFECTIVE (89% confidence)")
    pdf.ln(5)

    pdf.section_title("10. Evaluation Criteria Alignment")
    pdf.bullet("Detection Accuracy (35%): Multi-metric classical + quantum hybrid fusion")
    pdf.bullet("Innovation (20%): Full quantum vision pipeline with variational quantum classifier")
    pdf.bullet("Efficiency (20%): Fast CPU-only inference, no GPU required")
    pdf.bullet("Usability (15%): Drag-and-drop UI with heatmaps and downloadable reports")
    pdf.bullet("Presentation (10%): Visual analysis panels, circuit diagram, metrics dashboard")
    pdf.ln(5)

    pdf.section_title("11. Repository")
    pdf.body_text(
        "GitHub: https://github.com/Maheshwar422/qt-2.3-quantum-defect-detection\n"
        "Team: Maheshwar422"
    )

    pdf.output(OUTPUT)
    print(f"PDF saved: {os.path.abspath(OUTPUT)}")


if __name__ == "__main__":
    build_pdf()
