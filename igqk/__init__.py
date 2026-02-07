"""
IGQK - Information-Geometric Quantum Compression

The world's first neural network compression framework that uses
quantum mechanics on statistical manifolds.

Unique innovations:
- AutoIGQK: Automatic per-layer optimal compression discovery
- Quantum Entanglement: Cross-layer compression via mutual information
- Adaptive Annealing: Quantum temperature scheduling with phase detection
- Streaming Compression: Real-time adaptive precision during inference
- .igqk Format: Ultra-compact binary model format (2 bits/weight)
- One-line HuggingFace integration

Unified theory covering:
- HLWT (Hybrid Laplace-Wavelet Transformation)
- TLGT (Ternary Lie Group Theory)
- FCHL (Fractional Calculus for Hebbian Learning)
"""

__version__ = '2.0.0'
__author__ = 'IGQK Research Team'

from .core.manifold import StatisticalManifold
from .core.quantum_state import QuantumState
from .core.evolution import QuantumGradientFlow
from .core.measurement import MeasurementOperator
from .integration.pytorch import IGQKOptimizer, IGQKTrainer
from .compression.projection import OptimalProjection
from .auto import AutoIGQK
from .entanglement import QuantumEntanglementCompressor
from .annealing import QuantumAnnealingScheduler
from .streaming import StreamingAdaptiveModel
from .format import IGQKFormat
from .visualizer import QuantumVisualizer

__all__ = [
    # Core
    'StatisticalManifold',
    'QuantumState',
    'QuantumGradientFlow',
    'MeasurementOperator',
    'IGQKOptimizer',
    'IGQKTrainer',
    'OptimalProjection',
    # Innovations
    'AutoIGQK',
    'QuantumEntanglementCompressor',
    'QuantumAnnealingScheduler',
    'StreamingAdaptiveModel',
    'IGQKFormat',
    'QuantumVisualizer',
]
