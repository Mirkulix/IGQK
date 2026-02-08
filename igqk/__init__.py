"""
IGQK - Information-Geometric Quantum Compression

The world's first neural network compression framework that uses
quantum mechanics on statistical manifolds.

Core innovations:
- AutoIGQK: Automatic per-layer optimal compression discovery
- Quantum Entanglement: Cross-layer compression via mutual information
- Adaptive Annealing: Quantum temperature scheduling with phase detection
- Streaming Compression: Real-time adaptive precision during inference
- .igqk Format: Ultra-compact binary model format (2 bits/weight)
- One-line HuggingFace integration

Advanced features:
- Self-Healing Compression: Autonomous accuracy recovery in production
- Temporal Transformer Compression: Position-aware adaptive precision
- Interpretable Compression: Explain what compression removes and why
- Quantum Transfer Learning: Entropy-guided freeze/finetune decisions
- Hardware-Adaptive Compilation: One model, optimal deployment everywhere
- Multi-Objective Quantum Flow: Joint accuracy/compression/latency optimization
- Federated Quantum Compression: Privacy-preserving distributed compression
- Compression-Aware NAS: Search for inherently compressible architectures

Unified theory covering:
- HLWT (Hybrid Laplace-Wavelet Transformation)
- TLGT (Ternary Lie Group Theory)
- FCHL (Fractional Calculus for Hebbian Learning)
"""

__version__ = '5.0.0'
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
from .healing import SelfHealingModel
from .temporal import TemporalCompressor
from .interpretable import InterpretableCompressor
from .transfer import QuantumTransferLearning
from .hardware import HardwareAdaptiveCompiler
from .multiobjective import MultiObjectiveFlow
from .federated import FederatedDevice, FederatedCoordinator
from .nas import CompressionAwareNAS
from .export import ONNXExporter
from .metrics import CompressionMetrics
from .plugins import PluginRegistry
from .zoo import ModelZoo
from .meta_learner import MetaLearner
from .evolution_engine import EvolutionEngine
from .auto_discovery import AutoDiscovery
from .knowledge_transfer import KnowledgeTransfer
from .autonomous import AutonomousPipeline
from .consciousness import QuantumConsciousnessMonitor
from .dna import DNAExtractor, CompressionDNA
from .predictor import CompressionOracle
from .time_travel import TimeTravelDebugger

__all__ = [
    # Core
    'StatisticalManifold',
    'QuantumState',
    'QuantumGradientFlow',
    'MeasurementOperator',
    'IGQKOptimizer',
    'IGQKTrainer',
    'OptimalProjection',
    # Innovations v2
    'AutoIGQK',
    'QuantumEntanglementCompressor',
    'QuantumAnnealingScheduler',
    'StreamingAdaptiveModel',
    'IGQKFormat',
    'QuantumVisualizer',
    # Advanced v3
    'SelfHealingModel',
    'TemporalCompressor',
    'InterpretableCompressor',
    'QuantumTransferLearning',
    'HardwareAdaptiveCompiler',
    'MultiObjectiveFlow',
    'FederatedDevice',
    'FederatedCoordinator',
    'CompressionAwareNAS',
    # Production v3.1
    'ONNXExporter',
    'CompressionMetrics',
    'PluginRegistry',
    'ModelZoo',
    # Self-Evolving AI v4.0
    'MetaLearner',
    'EvolutionEngine',
    'AutoDiscovery',
    'KnowledgeTransfer',
    'AutonomousPipeline',
    # Future v5.0
    'QuantumConsciousnessMonitor',
    'DNAExtractor',
    'CompressionDNA',
    'CompressionOracle',
    'TimeTravelDebugger',
]
