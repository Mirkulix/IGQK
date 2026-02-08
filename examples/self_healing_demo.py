#!/usr/bin/env python3
"""
Example: Self-Healing Compression in Production.

Demonstrates how the SelfHealingModel autonomously
monitors and repairs compressed model accuracy.

Usage:
    python examples/self_healing_demo.py
"""

import torch
import torch.nn as nn
from igqk import SelfHealingModel


def main():
    print("=" * 60)
    print("IGQK Example: Self-Healing Compression")
    print("=" * 60)

    # Create and compress model
    model = nn.Sequential(
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    healing = SelfHealingModel(
        model,
        confidence_threshold=0.7,
        window_size=50,
        heal_cooldown=20,
    )

    print("\n--- Initial Compression ---")
    healing.compress_initial(method="ternary")
    report = healing.get_health_report()
    print(f"  Step: {report['step']}")
    print(f"  Healing events: {report['healing_events']}")

    # Simulate production inference
    print("\n--- Production Inference (100 batches) ---")
    for i in range(100):
        x = torch.randn(8, 32)
        output = healing(x)

        if (i + 1) % 25 == 0:
            report = healing.get_health_report()
            print(f"  Step {report['step']}: "
                  f"confidence={report['avg_confidence']}, "
                  f"healing_events={report['healing_events']}")

    # Final report
    print("\n--- Final Health Report ---")
    report = healing.get_health_report()
    print(f"  Steps:           {report['step']}")
    print(f"  Avg confidence:  {report['avg_confidence']}")
    print(f"  Healing events:  {report['healing_events']}")
    print(f"  Layers monitored: {len(report['layers'])}")

    for layer in report['layers'][:5]:
        print(f"    {layer['name']}: compression={layer['compression']}, "
              f"healed={layer['heal_count']}x")

    print("\nDone!")


if __name__ == "__main__":
    main()
