#!/usr/bin/env python3
"""
Example: Compression-Aware Neural Architecture Search.

Searches for neural network architectures that are inherently
more compressible, using quantum entropy as fitness criterion.

Usage:
    python examples/nas_search.py
"""

import torch
import torch.nn as nn
from igqk import CompressionAwareNAS


def train_fn(model, epochs):
    """Simple training function (random data for demo)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def eval_fn(model):
    """Simple evaluation function."""
    model.eval()
    with torch.no_grad():
        x = torch.randn(100, 784)
        out = model(x)
        probs = torch.softmax(out, dim=-1)
        confidence = probs.max(dim=-1).values.mean().item()
    return confidence


def main():
    print("=" * 60)
    print("IGQK Example: Compression-Aware NAS")
    print("=" * 60)

    nas = CompressionAwareNAS(
        in_features=784,
        num_classes=10,
        population_size=8,
        generations=5,
        min_layers=2,
        max_layers=5,
    )

    print(f"\n  Population: {nas.population_size}")
    print(f"  Generations: {nas.generations}")
    print(f"  Layers: {nas.min_layers}-{nas.max_layers}")

    print("\n--- Searching ---")
    best = nas.search(train_fn=train_fn, eval_fn=eval_fn, verbose=True)

    print(f"\n--- Best Architecture ---")
    print(f"  ID:              {best.id}")
    print(f"  Layer widths:    {best.layer_widths}")
    print(f"  Activations:     {best.activations}")
    print(f"  Skip connections:{best.skip_connections}")
    print(f"  Dropout rates:   {best.dropout_rates}")
    print(f"  Fitness:         {best.fitness:.4f}")
    print(f"  Accuracy:        {best.accuracy:.4f}")
    print(f"  Compressibility: {best.compressibility:.4f}")
    print(f"  Parameters:      {best.params:,}")

    # Build and test the best model
    model = nas.build_best_model()
    x = torch.randn(4, 784)
    out = model(x)
    print(f"\n  Output shape: {out.shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()
