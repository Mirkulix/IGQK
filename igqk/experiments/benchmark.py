"""
IGQK Compression Benchmark.

Compares IGQK compression against standard methods on common architectures.

Usage:
    python -m igqk.experiments.benchmark
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List

from igqk.core.quantum_state import QuantumState
from igqk.core.measurement import MeasurementOperator
from igqk.theory.tlgt import TernaryLieGroup
from igqk.theory.hlwt import HybridLaplaceWavelet
from igqk.compression.projection import OptimalProjection
from igqk.integration.pytorch import IGQKTrainer


def create_model(architecture: str, in_features: int, num_classes: int) -> nn.Module:
    """Create model by architecture name."""
    if architecture == "fc_small":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    elif architecture == "fc_medium":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    elif architecture == "fc_large":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def benchmark_compression(
    model: nn.Module,
    methods: List[str] = None,
) -> Dict[str, dict]:
    """
    Benchmark different compression methods on a model.

    Args:
        model: Trained PyTorch model.
        methods: List of methods to benchmark.

    Returns:
        Dict of method -> {compression_ratio, distortion, time_ms, sparsity}
    """
    if methods is None:
        methods = ["ternary", "wavelet", "sparse_10", "sparse_30"]

    results = {}
    original_params = torch.cat([p.flatten() for p in model.parameters()])
    original_size = original_params.numel()

    for method in methods:
        start = time.time()

        if method == "ternary":
            tlgt = TernaryLieGroup(original_size)
            compressed, scale = tlgt.quantize(original_params)
            stats = tlgt.compression_stats(original_params, compressed)
            results[method] = {
                "compression_ratio": stats["compression_ratio"],
                "distortion": stats["distortion"],
                "relative_error": stats["relative_error"],
                "sparsity": stats["sparsity"],
                "bits_per_weight": stats["bits_per_weight"],
            }

        elif method == "wavelet":
            hlwt = HybridLaplaceWavelet()
            compressed, ratio = hlwt.compress(original_params, keep_ratio=0.3)
            distortion = torch.norm(original_params - compressed).item() ** 2
            results[method] = {
                "compression_ratio": ratio,
                "distortion": distortion,
                "relative_error": distortion / (torch.norm(original_params).item() ** 2 + 1e-10),
                "sparsity": (compressed == 0).float().mean().item(),
                "bits_per_weight": 32 * ratio,
            }

        elif method.startswith("sparse_"):
            pct = int(method.split("_")[1]) / 100.0
            proj = OptimalProjection(submanifold_type="sparse", sparsity=pct)
            compressed = proj.projector.project(original_params)
            distortion = torch.norm(original_params - compressed).item() ** 2
            results[method] = {
                "compression_ratio": pct,
                "distortion": distortion,
                "relative_error": distortion / (torch.norm(original_params).item() ** 2 + 1e-10),
                "sparsity": (compressed == 0).float().mean().item(),
                "bits_per_weight": 32 * pct,
            }

        elapsed = (time.time() - start) * 1000
        results[method]["time_ms"] = elapsed

    return results


def run_full_benchmark():
    """Run full benchmark suite."""
    print("=" * 70)
    print("IGQK Compression Benchmark")
    print("=" * 70)

    architectures = ["fc_small", "fc_medium", "fc_large"]
    in_features = 784
    num_classes = 10

    for arch in architectures:
        model = create_model(arch, in_features, num_classes)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- {arch} ({total_params:,} params) ---")

        results = benchmark_compression(model)

        print(f"  {'Method':<15} {'Ratio':<10} {'Distortion':<12} {'Sparsity':<10} {'Time(ms)':<10}")
        print(f"  {'-'*57}")
        for method, stats in results.items():
            print(
                f"  {method:<15} "
                f"{stats['compression_ratio']:<10.4f} "
                f"{stats['distortion']:<12.4f} "
                f"{stats['sparsity']:<10.2%} "
                f"{stats['time_ms']:<10.1f}"
            )

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    run_full_benchmark()
