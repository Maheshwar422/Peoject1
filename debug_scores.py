import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2
from models.classical_model import classical_predict
from models.quantum_model import quantum_predict

for path in ["samples/good_product.png", "samples/defect_scratch.png", "samples/defect_texture.png"]:
    img = cv2.imread(str(ROOT / path))
    c = classical_predict(img)
    q = quantum_predict(c["features"]["feature_vector"])
    print(f"\n{path}")
    print(f"  metrics: {c['features']['metrics']}")
    print(f"  classical score: {c['defect_score']}, verdict: {c['verdict']}")
    print(f"  quantum prob: {q['defect_probability']}, score: {q['quantum_score']}, verdict: {q['verdict']}")
    print(f"  features: {c['features']['feature_vector']}")
