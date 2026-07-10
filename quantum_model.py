from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class QuantumAnalysisResult:
    confidence: float
    label: str
    state_probabilities: np.ndarray


class QuantumDefectClassifier:
    """Small quantum-inspired classifier for compressed feature vectors."""

    def __init__(self, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits

    def _build_circuit(self, features: np.ndarray) -> QuantumCircuit:
        clipped = np.clip(features[: self.num_qubits], 0.0, 1.0)
        qc = QuantumCircuit(self.num_qubits)

        for idx, value in enumerate(clipped):
            qc.ry(float(value * np.pi), idx)

        if self.num_qubits > 1:
            for idx in range(self.num_qubits - 1):
                qc.cx(idx, idx + 1)

        return qc

    def predict(self, features: np.ndarray) -> QuantumAnalysisResult:
        circuit = self._build_circuit(features)
        state = Statevector.from_instruction(circuit)
        probabilities = np.asarray(state.probabilities())

        defective_probability = float(probabilities[-1])
        confidence = float(np.clip(0.55 + defective_probability * 0.45, 0.0, 1.0))
        label = "Defective" if defective_probability >= 0.20 else "Likely OK"

        return QuantumAnalysisResult(
            confidence=confidence,
            label=label,
            state_probabilities=probabilities,
        )
