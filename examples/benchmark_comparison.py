#!/usr/bin/env python3
"""
Example: Benchmark comparison of all IGQK compression methods.

Compares ternary, wavelet, sparse, and auto compression on
multiple model architectures with full metrics.

Usage:
    python examples/benchmark_comparison.py
"""

import torch
import torch.nn as nn
import copy
import time
from igqk import AutoIGQK
from igqk.theory.tlgt import TernaryLieGroup
from igqk.theory.hlwt import HybridLaplaceWavelet
from igqk.compression.projection import OptimalProjection
from igqk.metrics import CompressionMetrics


def create_model(name):
    """Create test model by name."""
    if name == "small":
        return nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10),
        )
    elif name == "medium":
        return nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    elif name == "large":
        return nn.Sequential(
            nn.Linear(784, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )


def compress_ternary(model):
    compressed = copy.deepcopy(model)
    with torch.no_grad():
        for param in compressed.parameters():
            if param.numel() >= 16:
                tlgt = TernaryLieGroup(param.numel())
                c, s = tlgt.quantize(param.data)
                param.data = c
    return compressed


def compress_wavelet(model):
    compressed = copy.deepcopy(model)
    hlwt = HybridLaplaceWavelet()
    with torch.no_grad():
        for param in compressed.parameters():
            if param.numel() >= 16:
                c, r = hlwt.compress(param.data, keep_ratio=0.3)
                param.data = c
    return compressed


def compress_sparse(model, sparsity=0.1):
    compressed = copy.deepcopy(model)
    with torch.no_grad():
        for param in compressed.parameters():
            if param.numel() >= 16:
                threshold = torch.quantile(param.data.abs().flatten(), 1 - sparsity)
                param.data *= (param.data.abs() >= threshold).float()
    return compressed


def compress_auto(model):
    compressed = copy.deepcopy(model)
    auto = AutoIGQK(target_compression=0.1)
    auto.compress(compressed)
    return compressed


def main():
    print("=" * 80)
    print("IGQK Benchmark: Comparing All Compression Methods")
    print("=" * 80)

    metrics = CompressionMetrics()
    architectures = ["small", "medium", "large"]
    methods = {
        "Ternary": compress_ternary,
        "Wavelet": compress_wavelet,
        "Sparse-10%": lambda m: compress_sparse(m, 0.1),
        "Sparse-30%": lambda m: compress_sparse(m, 0.3),
        "AutoIGQK": compress_auto,
    }

    for arch_name in architectures:
        original = create_model(arch_name)
        total_params = sum(p.numel() for p in original.parameters())
        print(f"\n{'='*80}")
        print(f"Architecture: {arch_name} ({total_params:,} params)")
        print(f"{'='*80}")

        reports = []
        for method_name, compress_fn in methods.items():
            start = time.time()
            compressed = compress_fn(original)
            elapsed = (time.time() - start) * 1000

            report = metrics.evaluate(
                original, compressed,
                model_name=arch_name,
                method=method_name,
            )
            reports.append(report)

            print(f"\n  {method_name}:")
            print(f"    Compression: {report.compression_ratio:.1f}x")
            print(f"    Bits/weight: {report.bits_per_weight:.1f}")
            print(f"    Sparsity:    {report.sparsity:.1%}")
            print(f"    Weight PSNR: {report.weight_psnr:.1f} dB")
            print(f"    Cos sim:     {report.output_cosine_sim:.4f}")
            print(f"    Time:        {elapsed:.1f} ms")

        print(f"\n--- Comparison ---")
        metrics.compare(reports)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
