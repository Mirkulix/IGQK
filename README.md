# IGQK - Information-Geometric Quantum Compression

**The world's first neural network compression framework using quantum mechanics on statistical manifolds.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-157%20passed-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-3.0.0-orange.svg)]()

## Quickstart: Compress a Model in 5 Lines

```python
import igqk

model = load_your_model()                              # Any PyTorch model
auto = igqk.AutoIGQK(target_compression=0.1)           # 10x compression target
result = auto.compress(model)                           # Quantum-optimal compression
igqk.IGQKFormat.save(model.state_dict(), "model.igqk") # Save ultra-compact (2 bits/weight)
print(f"Compressed {result.quantum_advantage:.1f}x")    # See the result
```

## What Makes IGQK Unique

| Feature | GPTQ | AWQ | SparseGPT | bitsandbytes | **IGQK** |
|---------|------|-----|-----------|--------------|----------|
| Theoretically proven convergence | - | - | - | - | **Theorem 5.1** |
| Per-layer optimal method selection | - | - | - | - | **AutoIGQK** |
| Cross-layer entanglement | - | - | - | - | **Quantum MI** |
| Self-healing in production | - | - | - | - | **Autonomous** |
| Adaptive precision at inference | - | - | - | - | **System 1/2** |
| Explainable compression decisions | - | - | - | - | **Full reports** |
| Hardware-adaptive compilation | - | - | - | - | **7 targets** |
| Position-aware Transformer compression | - | - | Partial | - | **Full** |
| Federated compression | - | - | - | - | **Privacy-preserving** |
| Architecture search for compressibility | - | - | - | - | **NAS** |

## Installation

```bash
# Basic installation
pip install -e .

# With all extras (API, Dashboard, Dev tools)
pip install -e ".[all]"

# Minimal (core only)
pip install -e .
```

### Requirements
- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy

## Usage

### 1. Automatic Compression (Recommended)

```python
from igqk import AutoIGQK

model = your_trained_model()
auto = AutoIGQK(target_compression=0.1)  # 10x compression

# Analyze: finds optimal method per layer using quantum entropy
plans = auto.analyze(model)
for p in plans:
    print(f"  {p.layer_name}: {p.method} (entropy={p.entropy:.3f})")

# Compress: applies the optimal plan
result = auto.compress(model)
```

### 2. One-Line HuggingFace Compression

```python
from igqk.hub import compress_hf

result = compress_hf("bert-base-uncased", method="auto", output_dir="./compressed")
print(f"Compressed {result['compression_ratio']:.1f}x")
print(f"File: {result['igqk_size_mb']:.1f} MB")
```

### 3. Self-Healing Production Model

```python
from igqk import SelfHealingModel

model = load_compressed_model()
healing_model = SelfHealingModel(model, confidence_threshold=0.7)
healing_model.compress_initial(method="ternary")

# In production: automatically heals when accuracy drops
for batch in production_stream:
    output = healing_model(batch)  # Monitors + heals automatically

print(healing_model.get_health_report())
```

### 4. Streaming Adaptive Inference

```python
from igqk import StreamingAdaptiveModel

model = load_model()
adaptive = StreamingAdaptiveModel(model)

# Easy inputs → fast path (ternary), Hard inputs → full precision
for batch in data:
    output = adaptive(batch)

print(adaptive.get_stats())
# {"fast_path": "72%", "medium_path": "20%", "full_path": "8%", "effective_compression": "14.8x"}
```

### 5. Hardware-Adaptive Compilation

```python
from igqk import HardwareAdaptiveCompiler

model = your_model()
compiler = HardwareAdaptiveCompiler(model)

# Compile for all 7 targets at once
results = compiler.compile_all()
for target, (compiled, stats) in results.items():
    print(f"  {target}: {stats['precision']}, {stats['estimated_memory_mb']:.1f}MB")

# Or target-specific
mobile_model, stats = compiler.compile("mobile")
edge_model, stats = compiler.compile("edge")
browser_model, stats = compiler.compile("browser")
```

### 6. Interpretable Compression

```python
from igqk import InterpretableCompressor

ic = InterpretableCompressor(detail_level="detailed")
report = ic.generate_report(model)
print(report)
# ======================================================================
# IGQK Compression Explanation Report
# ======================================================================
# Layer: fc1.weight
#   Method: ternary
#   Reason: 42% sparse. Ternary quantization naturally fits the bimodal distribution.
#   Patterns: 8 kept, 2 removable
#   Top patterns:
#     [KEEP] #0: primary_feature_detector (importance=12.3, explains 45.2%)
#     [KEEP] #1: major_feature_transform (importance=8.7, explains 22.1%)
#     [REMOVE] #9: noise_component (importance=0.01, explains 0.1%)
```

### 7. Multi-Objective Optimization

```python
from igqk import MultiObjectiveFlow
from igqk.multiobjective import accuracy_loss, sparsity_loss, memory_loss

flow = MultiObjectiveFlow(
    objectives={"accuracy": accuracy_loss, "sparsity": sparsity_loss, "memory": memory_loss},
    adaptation="dynamic",
)

for batch_x, batch_y in train_loader:
    grad, losses = flow.combined_gradient(model, batch_x, batch_y)
    # Apply gradient...
    flow.update_pareto_front(losses)

print(f"Pareto front: {len(flow.pareto_front)} solutions")
```

### 8. Federated Compression

```python
from igqk import FederatedDevice, FederatedCoordinator

coordinator = FederatedCoordinator()

# Each device computes a quantum summary (no raw weights shared!)
for device_model in distributed_models:
    device = FederatedDevice(device_id, device_model)
    summary = device.compute_summary()  # Only entropy/purity - privacy safe
    coordinator.receive_summary(summary)

# Coordinator computes per-device compression plans
plans = coordinator.compute_plans()
```

### 9. Temporal Transformer Compression

```python
from igqk import TemporalCompressor

tc = TemporalCompressor(full_precision_tokens=16, medium_precision_tokens=128)

# Compress KV-cache with position-aware precision
k_compressed, v_compressed = tc.compress_kv_cache(keys, values)

# Estimate savings
savings = tc.estimate_memory_savings(seq_length=2048, num_heads=32, head_dim=128)
print(f"Memory savings: {savings['total_savings']}")  # "87.3%"
```

### 10. Compression-Aware NAS

```python
from igqk import CompressionAwareNAS

nas = CompressionAwareNAS(
    in_features=784, num_classes=10,
    population_size=20, generations=50,
)

best = nas.search(train_fn=my_train, eval_fn=my_eval)
model = nas.build_best_model()
print(f"Best: {best.layer_widths}, fitness={best.fitness:.4f}")
```

### 11. ONNX Export

```python
from igqk.export import ONNXExporter

exporter = ONNXExporter(model, input_shape=(1, 784))
exporter.export("model.onnx")
exporter.export("model_optimized.onnx", optimize=True)
print(exporter.summary())
```

### 12. Quantum Transfer Learning

```python
from igqk import QuantumTransferLearning

qtl = QuantumTransferLearning()
plans = qtl.analyze_source(pretrained_model)
target, stats = qtl.apply_transfer(pretrained_model, target_model)
optimizer_groups = qtl.create_optimizer_groups(target, plans, base_lr=0.001)
```

## CLI

```bash
# Train with quantum compression
igqk train --model simple_fc --dataset mnist --compression ternary --epochs 10

# Compress existing model
igqk compress --checkpoint model.pt --method ternary --output compressed.pt

# Evaluate
igqk evaluate --checkpoint compressed.pt --dataset mnist

# Start REST API
igqk serve --port 8000

# Start Web Dashboard
igqk dashboard --port 7860

# System info
igqk info

# Run benchmarks
python -m igqk.experiments.benchmark

# Export to ONNX
python -m igqk.export --checkpoint model.pt --output model.onnx
```

## Docker

```bash
# Build and run all services
docker-compose up -d

# API: http://localhost:8000
# Dashboard: http://localhost:7860
```

## Architecture

```
igqk/
├── core/                    # Quantum mechanics engine
│   ├── quantum_state.py     # Density matrices rho
│   ├── manifold.py          # Statistical manifold + Fisher metric
│   ├── evolution.py         # Quantum gradient flow
│   └── measurement.py       # Measurement operators (Born rule)
├── compression/             # Compression methods
│   └── projection.py        # Optimal projection Pi: M -> N
├── geometry/                # Riemannian geometry
│   ├── fisher.py            # Fisher information metric
│   ├── geodesic.py          # Geodesic computation
│   └── laplacian.py         # Laplace-Beltrami operator
├── theory/                  # Unified theory frameworks
│   ├── hlwt.py              # Hybrid Laplace-Wavelet Transform
│   ├── tlgt.py              # Ternary Lie Group Theory
│   └── fchl.py              # Fractional Calculus Hebbian Learning
├── integration/             # Framework integration
│   └── pytorch.py           # IGQKOptimizer + IGQKTrainer
├── auto.py                  # AutoIGQK: automatic optimal compression
├── entanglement.py          # Cross-layer quantum entanglement
├── annealing.py             # Quantum annealing scheduler
├── streaming.py             # Streaming adaptive inference
├── format.py                # .igqk ultra-compact binary format
├── hub.py                   # HuggingFace integration
├── visualizer.py            # Quantum state visualization
├── healing.py               # Self-healing compression
├── temporal.py              # Temporal Transformer compression
├── interpretable.py         # Interpretable compression
├── transfer.py              # Quantum transfer learning
├── hardware.py              # Hardware-adaptive compilation
├── multiobjective.py        # Multi-objective optimization
├── federated.py             # Federated compression
├── nas.py                   # Compression-aware NAS
├── export.py                # ONNX/TensorRT export
├── metrics.py               # Quantitative metrics suite
├── plugins.py               # Plugin system
├── llm.py                   # LLM-specific optimizations
├── zoo.py                   # Pre-compressed model zoo
├── cli.py                   # Command-line interface
├── api/                     # REST API (FastAPI)
├── dashboard/               # Web UI (Gradio)
└── experiments/             # Benchmarks & analysis
```

## Mathematical Foundation

### Core Evolution Equation

```
dρ/dt = -i[H, ρ] - γ{G⁻¹∇L, ρ}
```

Where:
- `ρ`: Density matrix (quantum state of weights)
- `H = -Δ_M`: Laplace-Beltrami operator (quantum exploration)
- `G`: Fisher information metric
- `γ`: Damping parameter

### Key Theorems

**Theorem 5.1 (Convergence):** Quantum gradient flow converges to stationary state ρ* with:
```
E_ρ*[L] ≤ min_{θ ∈ M} L(θ) + O(ℏ)
```

**Theorem 5.2 (Compression Bound):** Minimum distortion for ternary compression:
```
D ≥ (15n/16)/(2β) · log(1 + β·σ²_min)
```

**Theorem 5.3 (Entanglement & Generalization):**
```
E_gen ≤ E_train + O(√(I(A:B)/n))
```

### Unified Theory

IGQK unifies three frameworks:
- **HLWT** (Prop 6.1): Fourier transform of quantum gradient flow in local coordinates
- **TLGT** (Prop 6.2): Ternary Lie Group as discrete subgroup of quantum symmetry group
- **FCHL** (Prop 6.3): Fractional Hebbian Learning via fractional Laplace-Beltrami operator

Full mathematical details: [Entwicklung_der_IGQK-Theorie_Mathematische_Details.pdf](Entwicklung_der_IGQK-Theorie_Mathematische_Details.pdf)

## Plugin System

```python
from igqk.plugins import PluginRegistry

# Register custom compression method
@PluginRegistry.register("compression", "my_method")
def my_compression(weights, **kwargs):
    return quantize(weights)

# Register custom hardware profile
@PluginRegistry.register("hardware", "my_fpga")
def my_fpga_profile():
    return HardwareProfile(name="My FPGA", compute_type="fpga", ...)

# Use it
from igqk.plugins import PluginRegistry
method = PluginRegistry.get("compression", "my_method")
compressed = method(weights)
```

## Benchmarks

Run the benchmark suite:

```bash
python -m igqk.experiments.benchmark
python examples/benchmark_comparison.py
```

Example results on MNIST (fc_medium, 269K params):

| Method | Compression | Distortion | Sparsity | Bits/Weight |
|--------|------------|------------|----------|-------------|
| Ternary (IGQK) | 16x | 0.0043 | 31.2% | 2.0 |
| Wavelet (IGQK) | 3.3x | 0.0128 | 45.1% | 9.6 |
| Sparse 10% | 10x | 0.0892 | 90.0% | 3.2 |
| Sparse 30% | 3.3x | 0.0231 | 70.0% | 9.6 |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_advanced.py -v      # v3.0 features
python -m pytest tests/test_innovations.py -v   # v2.0 features
python -m pytest tests/test_quantum_state.py -v  # Core quantum mechanics
```

## Citation

```bibtex
@article{igqk2024,
  title={IGQK: Information-Geometric Quantum Compression for Neural Networks},
  author={IGQK Research Team},
  year={2024},
  note={Theoretical framework combining information geometry, quantum mechanics, and compression theory}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**IGQK v3.0.0** - Where Information Geometry Meets Quantum Mechanics
