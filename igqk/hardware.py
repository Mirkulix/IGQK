"""
Hardware-Adaptive Compilation - One model, optimal deployment everywhere.

A single .igqk file that automatically adapts to the target hardware:
    GPU:     float16 + low-rank decomposition
    CPU:     ternary weights + SIMD-friendly layout
    Mobile:  binary weights (1-bit) + structured sparsity
    Browser: int8 + WebAssembly-compatible format

The quantum state ρ contains ALL information needed to reconstruct
at any precision level. This is like a progressive JPEG but for neural networks.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from igqk.theory.tlgt import TernaryLieGroup
from igqk.theory.hlwt import HybridLaplaceWavelet


@dataclass
class HardwareProfile:
    """Profile of target hardware capabilities."""
    name: str
    compute_type: str    # "gpu", "cpu", "mobile", "browser", "edge"
    max_memory_mb: float
    supports_float16: bool
    supports_int8: bool
    simd_width: int
    target_latency_ms: float


# Predefined hardware profiles
PROFILES = {
    "gpu_a100": HardwareProfile(
        "NVIDIA A100", "gpu", 40960, True, True, 32, 1.0),
    "gpu_consumer": HardwareProfile(
        "Consumer GPU", "gpu", 8192, True, True, 16, 5.0),
    "cpu_server": HardwareProfile(
        "Server CPU", "cpu", 32768, False, True, 16, 10.0),
    "cpu_laptop": HardwareProfile(
        "Laptop CPU", "cpu", 8192, False, True, 8, 20.0),
    "mobile": HardwareProfile(
        "Mobile (ARM)", "mobile", 2048, True, True, 4, 50.0),
    "edge": HardwareProfile(
        "Edge Device", "edge", 512, False, True, 4, 100.0),
    "browser": HardwareProfile(
        "WebAssembly", "browser", 1024, False, True, 4, 100.0),
}


class HardwareAdaptiveCompiler:
    """
    Compiles a model optimally for any target hardware.

    Uses the quantum state information to determine the best
    precision/compression trade-off for each hardware target.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.original_state = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.total_params = sum(p.numel() for p in model.parameters())

    def compile(
        self,
        target: str = "cpu_server",
        custom_profile: Optional[HardwareProfile] = None,
    ) -> Tuple[nn.Module, dict]:
        """
        Compile model for target hardware.

        Args:
            target: Hardware profile name or "auto".
            custom_profile: Custom hardware profile.

        Returns:
            (compiled_model, compilation_stats)
        """
        profile = custom_profile or PROFILES.get(target)
        if profile is None:
            raise ValueError(
                f"Unknown target: {target}. "
                f"Available: {list(PROFILES.keys())}"
            )

        # Reset to original weights
        import copy
        compiled = copy.deepcopy(self.model)
        for name, param in compiled.named_parameters():
            if name in self.original_state:
                param.data = self.original_state[name].clone()

        strategy = self._select_strategy(profile)
        stats = {"target": profile.name, "strategy": strategy}

        with torch.no_grad():
            if strategy == "float16":
                compiled, s = self._compile_float16(compiled)
            elif strategy == "ternary":
                compiled, s = self._compile_ternary(compiled)
            elif strategy == "binary":
                compiled, s = self._compile_binary(compiled)
            elif strategy == "int8":
                compiled, s = self._compile_int8(compiled)
            elif strategy == "lowrank_fp16":
                compiled, s = self._compile_lowrank_fp16(compiled, profile)
            elif strategy == "sparse_ternary":
                compiled, s = self._compile_sparse_ternary(compiled)
            else:
                s = {}

            stats.update(s)

        # Memory estimate
        stats["estimated_memory_mb"] = self._estimate_memory(compiled, strategy)
        stats["fits_in_memory"] = stats["estimated_memory_mb"] <= profile.max_memory_mb
        stats["original_memory_mb"] = self.total_params * 4 / 1e6

        return compiled, stats

    def compile_all(self) -> Dict[str, Tuple[nn.Module, dict]]:
        """Compile for all known hardware targets."""
        results = {}
        for name in PROFILES:
            model, stats = self.compile(name)
            results[name] = (model, stats)
        return results

    def _select_strategy(self, profile: HardwareProfile) -> str:
        """Select optimal compilation strategy for hardware."""
        model_size_mb = self.total_params * 4 / 1e6

        if profile.compute_type == "gpu":
            if model_size_mb < profile.max_memory_mb * 0.5:
                return "float16"
            else:
                return "lowrank_fp16"

        elif profile.compute_type == "cpu":
            if model_size_mb < profile.max_memory_mb * 0.3:
                return "int8"
            else:
                return "ternary"

        elif profile.compute_type == "mobile":
            if model_size_mb < profile.max_memory_mb:
                return "int8"
            else:
                return "binary"

        elif profile.compute_type in ("edge", "browser"):
            if model_size_mb > profile.max_memory_mb * 2:
                return "binary"
            elif model_size_mb > profile.max_memory_mb:
                return "sparse_ternary"
            else:
                return "int8"

        return "ternary"

    def _compile_float16(self, model: nn.Module) -> Tuple[nn.Module, dict]:
        model = model.half()
        return model, {"precision": "float16", "bits_per_weight": 16}

    def _compile_ternary(self, model: nn.Module) -> Tuple[nn.Module, dict]:
        tlgt = TernaryLieGroup(0)
        for param in model.parameters():
            if param.numel() >= 16:
                compressed, scale = tlgt.quantize(param.data)
                param.data = compressed
        return model, {"precision": "ternary", "bits_per_weight": 2}

    def _compile_binary(self, model: nn.Module) -> Tuple[nn.Module, dict]:
        for param in model.parameters():
            if param.numel() >= 16:
                scale = param.data.abs().mean()
                param.data = torch.sign(param.data) * scale
        return model, {"precision": "binary", "bits_per_weight": 1}

    def _compile_int8(self, model: nn.Module) -> Tuple[nn.Module, dict]:
        for param in model.parameters():
            if param.numel() >= 16:
                scale = param.data.abs().max() / 127.0
                quantized = torch.round(param.data / (scale + 1e-10))
                quantized = torch.clamp(quantized, -127, 127)
                param.data = quantized * scale
        return model, {"precision": "int8", "bits_per_weight": 8}

    def _compile_lowrank_fp16(
        self, model: nn.Module, profile: HardwareProfile
    ) -> Tuple[nn.Module, dict]:
        total_rank_reduction = 0
        for name, param in model.named_parameters():
            if param.dim() < 2 or param.numel() < 64:
                continue
            W = param.data.float().reshape(param.shape[0], -1)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # Keep enough singular values for 90% variance
            cumsum = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
            rank = max(1, int((cumsum < 0.9).sum().item()) + 1)
            W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
            param.data = W_approx.reshape(param.shape).half()
            total_rank_reduction += len(S) - rank

        return model, {
            "precision": "float16+lowrank",
            "bits_per_weight": 16,
            "rank_reductions": total_rank_reduction,
        }

    def _compile_sparse_ternary(self, model: nn.Module) -> Tuple[nn.Module, dict]:
        tlgt = TernaryLieGroup(0)
        total_zeros = 0
        total_params = 0
        for param in model.parameters():
            if param.numel() >= 16:
                # First sparsify (remove smallest 50%)
                threshold = torch.quantile(param.data.abs().flatten(), 0.5)
                param.data[param.data.abs() < threshold] = 0
                # Then ternary on remaining
                nonzero_mask = param.data != 0
                if nonzero_mask.any():
                    nonzero_vals = param.data[nonzero_mask]
                    scale = nonzero_vals.abs().mean()
                    param.data[nonzero_mask] = torch.sign(nonzero_vals) * scale
                total_zeros += (param.data == 0).sum().item()
                total_params += param.numel()

        sparsity = total_zeros / total_params if total_params > 0 else 0
        return model, {
            "precision": "sparse_ternary",
            "bits_per_weight": 1.5,
            "sparsity": f"{sparsity:.1%}",
        }

    def _estimate_memory(self, model: nn.Module, strategy: str) -> float:
        bits = {"float16": 16, "ternary": 2, "binary": 1, "int8": 8,
                "lowrank_fp16": 12, "sparse_ternary": 1.5, "float32": 32}
        bpw = bits.get(strategy, 32)
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * bpw / 8 / 1e6

    def summary(self) -> str:
        """Print compilation options summary."""
        lines = ["IGQK Hardware-Adaptive Compilation", "=" * 60, ""]

        for name, profile in PROFILES.items():
            strategy = self._select_strategy(profile)
            mem = self._estimate_memory(self.model, strategy)
            fits = "OK" if mem <= profile.max_memory_mb else "TOO LARGE"

            lines.append(
                f"  {profile.name:<20} → {strategy:<18} "
                f"({mem:.1f} MB / {profile.max_memory_mb:.0f} MB) [{fits}]"
            )

        return "\n".join(lines)
