#!/usr/bin/env python3
"""
Example: Using the IGQK Plugin System.

Demonstrates how to register and use custom compression
methods, metrics, and hardware profiles.

Usage:
    python examples/plugin_example.py
"""

import torch
import torch.nn as nn
from igqk.plugins import PluginRegistry


def main():
    print("=" * 60)
    print("IGQK Example: Plugin System")
    print("=" * 60)

    # 1. List built-in plugins
    print("\n--- Built-in Plugins ---")
    plugins = PluginRegistry.list_plugins()
    for category, names in plugins.items():
        if names:
            print(f"  [{category}]: {', '.join(names)}")

    # 2. Register a custom compression method
    print("\n--- Register Custom Plugins ---")

    @PluginRegistry.register(
        "compression", "binary",
        description="Binary quantization to {-1, +1}",
        author="demo",
    )
    def binary_compress(weights, **kwargs):
        return torch.sign(weights) * weights.abs().mean()

    @PluginRegistry.register(
        "metric", "ternary_ratio",
        description="Fraction of weights that are ternary",
        author="demo",
    )
    def ternary_ratio(model, **kwargs):
        total = 0
        ternary = 0
        for p in model.parameters():
            flat = p.flatten()
            total += flat.numel()
            ternary += ((flat == -1) | (flat == 0) | (flat == 1)).sum().item()
        return ternary / total if total > 0 else 0

    print("  Registered: binary compression")
    print("  Registered: ternary_ratio metric")

    # 3. Use plugins
    print("\n--- Use Plugins ---")
    weights = torch.randn(100)

    # Built-in ternary
    ternary_fn = PluginRegistry.get("compression", "ternary_basic")
    ternary_weights = ternary_fn(weights)
    print(f"  Ternary:  unique values = {ternary_weights.unique().numel()}")

    # Built-in top-k sparse
    sparse_fn = PluginRegistry.get("compression", "top_k_sparse")
    sparse_weights = sparse_fn(weights, k=0.3)
    print(f"  Sparse:   nonzero = {(sparse_weights != 0).sum().item()}/{weights.numel()}")

    # Built-in uniform quantize
    quant_fn = PluginRegistry.get("compression", "uniform_quantize")
    quant_weights = quant_fn(weights, bits=4)
    print(f"  4-bit:    unique values = {quant_weights.unique().numel()}")

    # Custom binary
    binary_fn = PluginRegistry.get("compression", "binary")
    binary_weights = binary_fn(weights)
    print(f"  Binary:   unique values = {binary_weights.unique().numel()}")

    # 4. Plugin summary
    print("\n--- Plugin Summary ---")
    print(PluginRegistry.summary())


if __name__ == "__main__":
    main()
