#!/usr/bin/env python3
"""
Example: Federated Quantum Compression.

Demonstrates privacy-preserving distributed compression:
- Multiple devices compute quantum summaries (no raw weights shared)
- Central coordinator determines per-device compression plans
- Models are aggregated using federated averaging

Usage:
    python examples/federated_demo.py
"""

import torch
import torch.nn as nn
from igqk import FederatedDevice, FederatedCoordinator


def make_model(seed):
    """Create a model with different random weights (simulating different devices)."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def main():
    print("=" * 60)
    print("IGQK Example: Federated Quantum Compression")
    print("=" * 60)

    num_devices = 5
    coordinator = FederatedCoordinator()

    # Phase 1: Each device computes a quantum summary
    print("\n--- Phase 1: Quantum Summaries ---")
    devices = {}
    for i in range(num_devices):
        model = make_model(seed=42 + i)
        device = FederatedDevice(f"device_{i}", model)
        summary = device.compute_summary()
        coordinator.receive_summary(summary)
        devices[f"device_{i}"] = device

        print(f"  Device {i}: entropy={summary.global_entropy:.4f}, "
              f"purity={summary.global_purity:.4f}, "
              f"params={summary.num_params:,}")

    print(f"\n  Privacy: Only entropy/purity shared - NO raw weights!")

    # Phase 2: Coordinator computes per-device plans
    print("\n--- Phase 2: Compression Plans ---")
    plans = coordinator.compute_plans()
    for device_id, plan in plans.items():
        methods = set(plan.per_layer_method.values())
        print(f"  {device_id}: methods={methods}, "
              f"target={plan.target_compression:.2f}")

    # Phase 3: Global stats
    print("\n--- Phase 3: Global Statistics ---")
    stats = coordinator.get_global_stats()
    print(f"  Devices:          {stats['num_devices']}")
    print(f"  Avg entropy:      {stats['avg_global_entropy']:.4f}")
    print(f"  Avg purity:       {stats['avg_global_purity']:.4f}")
    print(f"  Entropy std:      {stats['entropy_std']:.4f}")

    # Phase 4: Aggregate models
    print("\n--- Phase 4: Federated Aggregation ---")
    state_dicts = {
        did: dev.get_state_dict()
        for did, dev in devices.items()
    }
    aggregated = coordinator.aggregate_models(state_dicts)
    print(f"  Aggregated {len(state_dicts)} models into global model")
    print(f"  Layers: {len(aggregated)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
