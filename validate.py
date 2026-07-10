"""Quick validation of the defect detection pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from models.hybrid_model import hybrid_predict

samples = [
    "samples/good_product.png",
    "samples/defect_scratch.png",
    "samples/defect_texture.png",
]

print("QT-2.3 Pipeline Validation\n" + "=" * 50)
for path in samples:
    img = cv2.imread(str(ROOT / path))
    result = hybrid_predict(img)
    final = result["final"]
    print(f"\n{path}")
    print(f"  Final: {final['status']} (confidence: {final['confidence']:.1%})")
    print(f"  Classical: {result['classical']['verdict']} | Quantum: {result['quantum']['verdict']}")
    print(f"  Agree: {final['models_agree']}")

print("\n" + "=" * 50)
print("Validation complete.")
