#!/usr/bin/env python3
"""
Example: Compress a ResNet model with IGQK.

Demonstrates:
- AutoIGQK for automatic per-layer optimal compression
- Interpretable compression reports
- Hardware-adaptive compilation
- Metrics evaluation

Usage:
    python examples/compress_resnet.py
"""

import torch
import torch.nn as nn
from igqk import AutoIGQK, InterpretableCompressor, HardwareAdaptiveCompiler
from igqk.metrics import CompressionMetrics


def create_resnet_like():
    """Create a simplified ResNet-like model for demonstration."""
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def main():
    print("=" * 60)
    print("IGQK Example: Compress ResNet-like Model")
    print("=" * 60)

    # 1. Create model
    model = create_resnet_like()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")

    # 2. AutoIGQK compression
    print("\n--- AutoIGQK Analysis ---")
    auto = AutoIGQK(target_compression=0.1)
    plans = auto.analyze(model)
    for p in plans:
        print(f"  {p.layer_name}: {p.method} (entropy={p.entropy:.3f}, strength={p.strength:.2f})")

    print("\n--- Compressing ---")
    import copy
    original = copy.deepcopy(model)
    result = auto.compress(model)
    print(f"  Quantum advantage: {result.quantum_advantage:.2f}x")

    # 3. Interpretable report
    print("\n--- Compression Explanation ---")
    ic = InterpretableCompressor(detail_level="medium")
    report = ic.generate_report(model)
    print(report)

    # 4. Hardware compilation
    print("\n--- Hardware-Adaptive Compilation ---")
    compiler = HardwareAdaptiveCompiler(model)
    print(compiler.summary())

    # 5. Metrics
    print("\n--- Metrics Evaluation ---")
    metrics = CompressionMetrics()
    eval_report = metrics.evaluate(original, model, model_name="ResNet-like", method="AutoIGQK")
    metrics.print_report(eval_report)


if __name__ == "__main__":
    main()
