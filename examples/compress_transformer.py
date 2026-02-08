#!/usr/bin/env python3
"""
Example: Compress a Transformer model with IGQK.

Demonstrates:
- Temporal compression (position-aware precision)
- KV-cache compression
- Attention head analysis
- Memory savings estimation

Usage:
    python examples/compress_transformer.py
"""

import torch
import torch.nn as nn
from igqk import TemporalCompressor
from igqk.llm import KVCacheCompressor, KVCacheConfig


def main():
    print("=" * 60)
    print("IGQK Example: Transformer Compression")
    print("=" * 60)

    # 1. Temporal Compression
    print("\n--- Temporal Compression ---")
    tc = TemporalCompressor(
        full_precision_tokens=16,
        medium_precision_tokens=128,
        head_prune_ratio=0.25,
    )

    # Create position mask
    seq_length = 512
    mask = tc.create_position_mask(seq_length)
    full = (mask == 1.0).sum().item()
    medium = (mask == 0.5).sum().item()
    heavy = (mask == 0.1).sum().item()
    print(f"  Sequence length: {seq_length}")
    print(f"  Full precision:  {full} tokens ({100*full/seq_length:.0f}%)")
    print(f"  Medium (int8):   {medium} tokens ({100*medium/seq_length:.0f}%)")
    print(f"  Ternary (2-bit): {heavy} tokens ({100*heavy/seq_length:.0f}%)")

    # 2. KV-Cache Compression
    print("\n--- KV-Cache Compression ---")
    batch, heads, head_dim = 1, 12, 64
    keys = torch.randn(batch, heads, seq_length, head_dim)
    values = torch.randn(batch, heads, seq_length, head_dim)

    k_comp, v_comp = tc.compress_kv_cache(keys, values)
    k_diff = (keys - k_comp).norm().item()
    v_diff = (values - v_comp).norm().item()
    print(f"  Key distortion:   {k_diff:.4f}")
    print(f"  Value distortion: {v_diff:.4f}")

    # 3. Memory Savings
    print("\n--- Memory Savings ---")
    savings = tc.estimate_memory_savings(
        seq_length=2048, num_heads=32, head_dim=128, batch_size=1
    )
    print(f"  Original:          {savings['original_mb']:.1f} MB")
    print(f"  After temporal:    {savings['after_temporal_mb']:.1f} MB ({savings['temporal_savings']})")
    print(f"  After head prune:  {savings['after_pruning_mb']:.1f} MB ({savings['total_savings']})")
    print(f"  Pruned heads:      {savings['pruned_heads']}")

    # 4. Streaming KV-Cache
    print("\n--- Streaming KV-Cache ---")
    config = KVCacheConfig(max_cache_size=256, compression_after=64)
    kv_compressor = KVCacheCompressor(config)

    for t in range(100):
        new_k = torch.randn(1, 4, 1, 32)
        new_v = torch.randn(1, 4, 1, 32)
        full_k, full_v = kv_compressor.update(new_k, new_v)

    stats = kv_compressor.stats()
    print(f"  Cache size:     {stats['size']} tokens")
    print(f"  Memory:         {stats['memory_mb']:.2f} MB")
    print(f"  Avg importance: {stats['avg_importance']:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
