"""
Quantitative Metrics Suite for IGQK compression evaluation.

Comprehensive metrics for comparing compression quality:
- Accuracy metrics (top-1, top-5, per-class)
- Compression metrics (ratio, bits/weight, sparsity)
- Performance metrics (latency, throughput, memory)
- Quality metrics (distortion, PSNR, fidelity)
- Quantum metrics (entropy, purity, entanglement)

Usage:
    from igqk.metrics import CompressionMetrics
    metrics = CompressionMetrics()
    report = metrics.evaluate(original_model, compressed_model, test_loader)
    metrics.print_report(report)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field


@dataclass
class MetricsReport:
    """Complete evaluation report."""
    model_name: str = "unknown"
    compression_method: str = "unknown"

    # Accuracy
    original_accuracy: float = 0.0
    compressed_accuracy: float = 0.0
    accuracy_drop: float = 0.0
    top5_original: float = 0.0
    top5_compressed: float = 0.0

    # Compression
    original_params: int = 0
    compressed_params: int = 0
    compression_ratio: float = 1.0
    bits_per_weight: float = 32.0
    sparsity: float = 0.0
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0

    # Performance
    original_latency_ms: float = 0.0
    compressed_latency_ms: float = 0.0
    speedup: float = 1.0
    original_throughput: float = 0.0
    compressed_throughput: float = 0.0

    # Quality
    weight_distortion: float = 0.0
    weight_psnr: float = 0.0
    output_mse: float = 0.0
    output_cosine_sim: float = 0.0

    # Quantum
    avg_entropy: float = 0.0
    avg_purity: float = 0.0

    # Per-layer details
    layer_details: List[Dict[str, Any]] = field(default_factory=list)


class CompressionMetrics:
    """Comprehensive metrics suite for compression evaluation."""

    def __init__(self, device: str = "cpu", num_warmup: int = 5, num_measure: int = 20):
        self.device = device
        self.num_warmup = num_warmup
        self.num_measure = num_measure

    def evaluate(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_loader=None,
        model_name: str = "model",
        method: str = "igqk",
    ) -> MetricsReport:
        """Run full evaluation suite."""
        report = MetricsReport(model_name=model_name, compression_method=method)

        # Compression metrics
        self._compute_compression_metrics(original_model, compressed_model, report)

        # Quality metrics
        self._compute_quality_metrics(original_model, compressed_model, report)

        # Accuracy metrics (if test loader provided)
        if test_loader is not None:
            self._compute_accuracy_metrics(original_model, compressed_model, test_loader, report)

        # Performance metrics
        self._compute_performance_metrics(original_model, compressed_model, report)

        # Quantum metrics
        self._compute_quantum_metrics(compressed_model, report)

        # Per-layer analysis
        self._compute_layer_details(original_model, compressed_model, report)

        return report

    def _compute_compression_metrics(
        self, original: nn.Module, compressed: nn.Module, report: MetricsReport
    ):
        """Compute compression ratio, sparsity, bits/weight."""
        orig_params = sum(p.numel() for p in original.parameters())
        comp_params = sum(p.numel() for p in compressed.parameters())

        orig_size = sum(p.numel() * p.element_size() for p in original.parameters())
        comp_nonzero = sum((p != 0).sum().item() for p in compressed.parameters())
        comp_zero = sum((p == 0).sum().item() for p in compressed.parameters())

        # Estimate effective compressed size
        effective_size = 0
        for p in compressed.parameters():
            flat = p.detach().flatten()
            unique_vals = flat.unique()
            if len(unique_vals) <= 4:
                effective_size += p.numel() * 2 // 8  # 2 bits
            elif (flat == 0).float().mean() > 0.7:
                effective_size += comp_nonzero * 6  # sparse: index + value
            else:
                effective_size += p.numel() * p.element_size()

        effective_size = max(effective_size, 1)

        report.original_params = orig_params
        report.compressed_params = comp_params
        report.original_size_mb = orig_size / 1e6
        report.compressed_size_mb = effective_size / 1e6
        report.compression_ratio = orig_size / effective_size
        report.sparsity = comp_zero / comp_params if comp_params > 0 else 0
        report.bits_per_weight = (effective_size * 8) / comp_params if comp_params > 0 else 32

    def _compute_quality_metrics(
        self, original: nn.Module, compressed: nn.Module, report: MetricsReport
    ):
        """Compute weight distortion, PSNR, output similarity."""
        orig_params = torch.cat([p.detach().flatten() for p in original.parameters()])
        comp_params = torch.cat([p.detach().flatten() for p in compressed.parameters()])

        # Weight-level metrics
        diff = orig_params - comp_params
        mse = (diff ** 2).mean().item()
        report.weight_distortion = mse

        if mse > 0:
            max_val = orig_params.abs().max().item()
            report.weight_psnr = 10 * np.log10((max_val ** 2) / mse)
        else:
            report.weight_psnr = float("inf")

        # Output-level metrics (random input)
        original.eval()
        compressed.eval()

        # Try to infer input shape from first layer
        first_param = list(original.parameters())[0]
        if first_param.dim() >= 2:
            in_features = first_param.shape[1]
        else:
            in_features = first_param.shape[0]

        with torch.no_grad():
            x = torch.randn(32, in_features)
            try:
                orig_out = original(x)
                comp_out = compressed(x)
                report.output_mse = ((orig_out - comp_out) ** 2).mean().item()
                # Cosine similarity
                cos_sim = nn.functional.cosine_similarity(
                    orig_out.flatten().unsqueeze(0),
                    comp_out.flatten().unsqueeze(0),
                ).item()
                report.output_cosine_sim = cos_sim
            except Exception:
                pass

    def _compute_accuracy_metrics(
        self, original: nn.Module, compressed: nn.Module,
        test_loader, report: MetricsReport
    ):
        """Compute top-1 and top-5 accuracy."""
        original.eval()
        compressed.eval()

        orig_correct1, orig_correct5 = 0, 0
        comp_correct1, comp_correct5 = 0, 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = targets.size(0)
                total += batch_size

                # Original
                orig_out = original(inputs)
                orig_top1 = orig_out.argmax(dim=1)
                orig_correct1 += (orig_top1 == targets).sum().item()
                if orig_out.shape[1] >= 5:
                    orig_top5 = orig_out.topk(5, dim=1).indices
                    orig_correct5 += sum(
                        targets[i] in orig_top5[i] for i in range(batch_size)
                    )

                # Compressed
                comp_out = compressed(inputs)
                comp_top1 = comp_out.argmax(dim=1)
                comp_correct1 += (comp_top1 == targets).sum().item()
                if comp_out.shape[1] >= 5:
                    comp_top5 = comp_out.topk(5, dim=1).indices
                    comp_correct5 += sum(
                        targets[i] in comp_top5[i] for i in range(batch_size)
                    )

        report.original_accuracy = orig_correct1 / total if total > 0 else 0
        report.compressed_accuracy = comp_correct1 / total if total > 0 else 0
        report.accuracy_drop = report.original_accuracy - report.compressed_accuracy
        report.top5_original = orig_correct5 / total if total > 0 else 0
        report.top5_compressed = comp_correct5 / total if total > 0 else 0

    def _compute_performance_metrics(
        self, original: nn.Module, compressed: nn.Module, report: MetricsReport
    ):
        """Measure latency and throughput."""
        first_param = list(original.parameters())[0]
        if first_param.dim() >= 2:
            in_features = first_param.shape[1]
        else:
            in_features = first_param.shape[0]

        x = torch.randn(1, in_features, device=self.device)
        original.eval()
        compressed.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup):
                original(x)
                compressed(x)

        # Measure original
        orig_times = []
        with torch.no_grad():
            for _ in range(self.num_measure):
                start = time.perf_counter()
                original(x)
                orig_times.append((time.perf_counter() - start) * 1000)

        # Measure compressed
        comp_times = []
        with torch.no_grad():
            for _ in range(self.num_measure):
                start = time.perf_counter()
                compressed(x)
                comp_times.append((time.perf_counter() - start) * 1000)

        report.original_latency_ms = np.median(orig_times)
        report.compressed_latency_ms = np.median(comp_times)
        report.speedup = report.original_latency_ms / max(report.compressed_latency_ms, 1e-6)
        report.original_throughput = 1000 / report.original_latency_ms if report.original_latency_ms > 0 else 0
        report.compressed_throughput = 1000 / report.compressed_latency_ms if report.compressed_latency_ms > 0 else 0

    def _compute_quantum_metrics(self, compressed: nn.Module, report: MetricsReport):
        """Compute quantum-inspired metrics (entropy, purity)."""
        from igqk.core.quantum_state import QuantumState

        entropies = []
        purities = []

        for param in compressed.parameters():
            if param.numel() < 16:
                continue
            rho = QuantumState.from_point(param.detach().flatten(), rank=min(8, param.numel()))
            entropies.append(rho.entropy())
            purities.append(rho.purity())

        report.avg_entropy = np.mean(entropies) if entropies else 0
        report.avg_purity = np.mean(purities) if purities else 0

    def _compute_layer_details(
        self, original: nn.Module, compressed: nn.Module, report: MetricsReport
    ):
        """Per-layer analysis."""
        orig_dict = dict(original.named_parameters())
        comp_dict = dict(compressed.named_parameters())

        for name in orig_dict:
            if name not in comp_dict:
                continue
            orig_p = orig_dict[name].detach()
            comp_p = comp_dict[name].detach()

            diff = (orig_p - comp_p).flatten()
            mse = (diff ** 2).mean().item()
            sparsity = (comp_p == 0).float().mean().item()
            unique = comp_p.flatten().unique().numel()

            report.layer_details.append({
                "name": name,
                "shape": list(orig_p.shape),
                "params": orig_p.numel(),
                "mse": mse,
                "sparsity": sparsity,
                "unique_values": unique,
                "is_ternary": unique <= 4,
            })

    def print_report(self, report: MetricsReport) -> str:
        """Generate formatted report string."""
        lines = [
            "=" * 70,
            f"IGQK Compression Metrics Report: {report.model_name}",
            f"Method: {report.compression_method}",
            "=" * 70,
            "",
            "--- Compression ---",
            f"  Parameters:       {report.original_params:,} -> {report.compressed_params:,}",
            f"  Size:             {report.original_size_mb:.2f} MB -> {report.compressed_size_mb:.2f} MB",
            f"  Compression:      {report.compression_ratio:.1f}x",
            f"  Bits/weight:      {report.bits_per_weight:.1f}",
            f"  Sparsity:         {report.sparsity:.1%}",
            "",
            "--- Accuracy ---",
            f"  Original top-1:   {report.original_accuracy:.4f}",
            f"  Compressed top-1: {report.compressed_accuracy:.4f}",
            f"  Accuracy drop:    {report.accuracy_drop:.4f}",
            "",
            "--- Quality ---",
            f"  Weight MSE:       {report.weight_distortion:.6f}",
            f"  Weight PSNR:      {report.weight_psnr:.2f} dB",
            f"  Output MSE:       {report.output_mse:.6f}",
            f"  Output cosine:    {report.output_cosine_sim:.6f}",
            "",
            "--- Performance ---",
            f"  Latency (orig):   {report.original_latency_ms:.2f} ms",
            f"  Latency (comp):   {report.compressed_latency_ms:.2f} ms",
            f"  Speedup:          {report.speedup:.2f}x",
            "",
            "--- Quantum ---",
            f"  Avg entropy:      {report.avg_entropy:.4f}",
            f"  Avg purity:       {report.avg_purity:.4f}",
            "",
        ]

        if report.layer_details:
            lines.append("--- Per-Layer ---")
            for ld in report.layer_details:
                ternary_flag = " [TERNARY]" if ld["is_ternary"] else ""
                lines.append(
                    f"  {ld['name']}: {ld['params']:,} params, "
                    f"MSE={ld['mse']:.6f}, sparsity={ld['sparsity']:.1%}{ternary_flag}"
                )

        lines.append("=" * 70)
        text = "\n".join(lines)
        print(text)
        return text

    def compare(self, reports: List[MetricsReport]) -> str:
        """Compare multiple compression results side by side."""
        if not reports:
            return "No reports to compare."

        header = f"{'Method':<20} {'Ratio':<8} {'Acc Drop':<10} {'Bits/W':<8} {'Sparsity':<10} {'Latency':<10} {'PSNR':<8}"
        lines = [
            "IGQK Compression Comparison",
            "=" * 80,
            header,
            "-" * 80,
        ]

        for r in reports:
            lines.append(
                f"{r.compression_method:<20} "
                f"{r.compression_ratio:<8.1f} "
                f"{r.accuracy_drop:<10.4f} "
                f"{r.bits_per_weight:<8.1f} "
                f"{r.sparsity:<10.1%} "
                f"{r.compressed_latency_ms:<10.2f} "
                f"{r.weight_psnr:<8.2f}"
            )

        lines.append("=" * 80)
        text = "\n".join(lines)
        print(text)
        return text
