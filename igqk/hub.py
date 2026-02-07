"""
HuggingFace Hub Integration - One-Line Compression.

Compress any HuggingFace model with a single function call:

    from igqk import compress_hf
    compressed = compress_hf("bert-base-uncased", method="ternary")
    compressed.push_to_hub("my-compressed-bert")

This is the fastest path from "I have a model" to "I have a compressed model".
No configuration, no training loop, no expertise needed.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path

from igqk.auto import AutoIGQK
from igqk.entanglement import QuantumEntanglementCompressor
from igqk.theory.tlgt import TernaryLieGroup
from igqk.theory.hlwt import HybridLaplaceWavelet
from igqk.format import IGQKFormat


def compress_hf(
    model_name_or_path: str,
    method: str = "auto",
    target_compression: float = 0.1,
    output_dir: Optional[str] = None,
    save_igqk: bool = True,
) -> Dict[str, Any]:
    """
    Compress any HuggingFace model with one line.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        method: "auto", "ternary", "wavelet", "entangled", "sparse".
        target_compression: Target compression ratio (0.1 = 10x).
        output_dir: Directory for compressed model.
        save_igqk: Also save in .igqk ultra-compact format.

    Returns:
        Dict with compressed model, stats, and paths.

    Example:
        result = compress_hf("bert-base-uncased")
        print(f"Compressed {result['compression_ratio']:.1f}x")
        print(f"File size: {result['igqk_size_mb']:.1f} MB")
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "HuggingFace transformers required: pip install transformers"
        )

    print(f"[IGQK] Loading {model_name_or_path}...")
    model = AutoModel.from_pretrained(model_name_or_path, torchscript=False)
    model.eval()

    original_params = sum(p.numel() for p in model.parameters())
    original_size = original_params * 4  # float32

    print(f"[IGQK] Model: {original_params:,} params ({original_size / 1e6:.1f} MB)")
    print(f"[IGQK] Compressing with method='{method}'...")

    if method == "auto":
        auto = AutoIGQK(target_compression=target_compression)
        result = auto.compress(model)
        stats = {
            "method": "auto",
            "plans": [
                {"layer": p.layer_name, "method": p.method, "strength": p.strength}
                for p in result.plans
            ],
            "quantum_advantage": result.quantum_advantage,
        }

    elif method == "entangled":
        compressor = QuantumEntanglementCompressor()
        model, pairs, ent_stats = compressor.compress_entangled(model)
        stats = {
            "method": "entangled",
            "entangled_pairs": ent_stats["entangled_pairs"],
            "compression_ratio": ent_stats["compression_ratio"],
        }

    elif method == "ternary":
        tlgt = TernaryLieGroup(dim=0)
        with torch.no_grad():
            for param in model.parameters():
                if param.numel() >= 16:
                    compressed, scale = tlgt.quantize(param.data)
                    param.data = compressed
        stats = {"method": "ternary"}

    elif method == "wavelet":
        hlwt = HybridLaplaceWavelet()
        with torch.no_grad():
            for param in model.parameters():
                if param.numel() >= 16:
                    compressed, ratio = hlwt.compress(param.data, keep_ratio=0.3)
                    param.data = compressed
        stats = {"method": "wavelet"}

    elif method == "sparse":
        with torch.no_grad():
            for param in model.parameters():
                if param.numel() >= 16:
                    threshold = torch.quantile(param.data.abs().flatten(), 0.7)
                    mask = param.data.abs() >= threshold
                    param.data *= mask.float()
        stats = {"method": "sparse"}

    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate compressed stats
    compressed_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    effective_size = _estimate_compressed_size(model)

    stats.update({
        "original_params": original_params,
        "original_size_mb": original_size / 1e6,
        "compressed_size_mb": effective_size / 1e6,
        "compression_ratio": original_size / effective_size if effective_size > 0 else 1,
        "sparsity": zero_params / compressed_params,
    })

    # Save
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save PyTorch format
        pt_path = output_path / "compressed_model.pt"
        torch.save(model.state_dict(), str(pt_path))
        stats["pt_path"] = str(pt_path)
        stats["pt_size_mb"] = pt_path.stat().st_size / 1e6

        # Save ultra-compact .igqk format
        if save_igqk:
            igqk_path = output_path / "model.igqk"
            file_size = IGQKFormat.save(
                model.state_dict(),
                str(igqk_path),
                metadata={
                    "source_model": model_name_or_path,
                    "method": method,
                    **{k: v for k, v in stats.items() if isinstance(v, (int, float, str))},
                },
            )
            stats["igqk_path"] = str(igqk_path)
            stats["igqk_size_mb"] = file_size / 1e6

    print(f"[IGQK] Done! {stats['compression_ratio']:.1f}x compression")
    print(f"[IGQK] Original:   {stats['original_size_mb']:.1f} MB")
    print(f"[IGQK] Compressed: {stats['compressed_size_mb']:.1f} MB")
    print(f"[IGQK] Sparsity:   {stats['sparsity']:.1%}")

    stats["model"] = model
    return stats


def _estimate_compressed_size(model: nn.Module) -> int:
    """Estimate compressed storage size based on weight distribution."""
    total = 0
    for param in model.parameters():
        flat = param.detach().flatten()
        unique = flat.unique()

        if len(unique) <= 4:
            # Ternary: 2 bits per weight
            total += param.numel() * 2 // 8
        elif (flat == 0).float().mean() > 0.7:
            # Sparse: store only nonzero
            nonzero = (flat != 0).sum().item()
            total += nonzero * 6  # index + value
        else:
            total += param.numel() * 4  # float32

    return max(total, 1)
