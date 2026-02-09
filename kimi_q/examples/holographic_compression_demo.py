"""
KIMI-Q Holographic Compression Demo

Demonstriert den vollständigen KIMI-Q Pipeline auf einem kleinen Netzwerk:

1. Training mit holographischem Ricci-Fluss
2. Topologische Analyse (Dirac-Operator, Atiyah-Singer-Index)
3. Holographische Kompression (Ryu-Takayanagi)
4. Verschränkungs-Pruning (Bell-Paare)
5. Zero-Shot Revival
6. Quantum-Lift & Squeezing

Verwendung:
    python -m kimi_q.examples.holographic_compression_demo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def create_toy_dataset(n_samples: int = 1000, n_features: int = 20, n_classes: int = 4):
    """Erzeuge einen Spielzeugdatensatz mit nicht-trivialer Topologie."""
    X = torch.randn(n_samples, n_features)
    # Nicht-lineare Entscheidungsgrenzen (erfordern tiefes Netz)
    angles = torch.atan2(X[:, 0], X[:, 1])
    radii = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = ((angles / (2 * np.pi) * n_classes + radii) % n_classes).long()
    return X, y


class ToyNet(nn.Module):
    """Kleines Netzwerk für die Demo."""
    def __init__(self, n_features: int = 20, n_hidden: int = 64, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def demo_individual_components():
    """Demonstriere jede KIMI-Q Komponente einzeln."""
    from kimi_q.core.kahler_manifold import KahlerManifold
    from kimi_q.core.ricci_flow import HolographicRicciFlow
    from kimi_q.core.entanglement_pruning import EntanglementPruner
    from kimi_q.core.holographic import HolographicEntropy
    from kimi_q.core.dirac_operator import DiracOperator
    from kimi_q.core.quantum_lift import QuantumLift

    print("=" * 60)
    print("KIMI-Q: Individuelle Komponenten Demo")
    print("=" * 60)

    model = ToyNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModell: {n_params} Parameter")

    X, y = create_toy_dataset(200)

    # --- 1. Kähler-Mannigfaltigkeit ---
    print("\n--- 1. Kähler-Mannigfaltigkeit ---")
    manifold = KahlerManifold(n_params=n_params, hbar=0.1)

    params = torch.cat([p.flatten() for p in model.parameters()])
    z = manifold.complexify(params)
    print(f"  Komplexifiziert: {z.shape} Parameter in C^n")
    print(f"  |z| = {z.abs().mean():.4f}")

    # Unsicherheitsrelation
    bounds = manifold.uncertainty_relation()
    print(f"  Heisenberg-Grenze: min = {bounds.min():.6f}")

    # --- 2. Holographische Entropie ---
    print("\n--- 2. Holographische Entropie (Ryu-Takayanagi) ---")
    holo = HolographicEntropy(bulk_dimension=3)

    rt = holo.ryu_takayanagi_entropy(model)
    for name, info in rt.items():
        print(f"  {name}: S_RT = {info['ryu_takayanagi_entropy']:.2f}, "
              f"eff_rank = {info['effective_rank']:.1f}/{info['full_rank']}")

    bounds = holo.holographic_compression_bound(n_params)
    print(f"  Holographische Grenze: {bounds['max_compression_ratio']:.1f}x Kompression")
    print(f"  Effektive Parameter: {bounds['effective_params']:.0f} (von {n_params})")

    # --- 3. Dirac-Operator & Topologie ---
    print("\n--- 3. Dirac-Operator & Topologische Invarianten ---")
    dirac = DiracOperator(higgs_mass=1.0)

    index_info = dirac.atiyah_singer_index(model)
    print(f"  Atiyah-Singer-Index: {index_info['total_index']}")
    print(f"  Topologisches Budget: {index_info['topological_budget']} Parameter")
    print(f"  Überschüssige Parameter: {index_info['excess_params']}")
    print(f"  Max Kompression (Topologie): {index_info['max_compression_by_topology']:.1f}x")

    spectral_action = dirac.spectral_action(model)
    print(f"  Spektrale Aktion S[D]: {spectral_action:.2f}")

    # --- 4. Verschränkungs-Pruning ---
    print("\n--- 4. Verschränkungs-Pruning ---")
    pruner = EntanglementPruner(entropy_threshold=0.1)

    # Verschränkungskarte
    ent_map = pruner.compute_entanglement_map(model)
    for key, entropies in ent_map.items():
        print(f"  {key}: mean S = {entropies.mean():.4f}, "
              f"prunbar = {(entropies < 0.1).float().mean():.1%}")

    # Pruning
    pre_params = sum(p.numel() for p in model.parameters())
    stats = pruner.prune_with_entanglement(model, target_sparsity=0.3)
    print(f"  Gepruned: {stats['pruned']} / {stats['total_params']} Parameter")
    print(f"  Bell-Paare: {stats['bell_pairs_created']}")
    print(f"  Sparsity: {stats['actual_sparsity']:.2%}")
    print(f"  Kompression: {stats['compression_ratio']:.2f}x")

    # Zero-Shot Revival
    revival_stats = pruner.revival(model)
    print(f"  Revival: {revival_stats['revived']} Neuronen reaktiviert")

    # --- 5. Quantum Lift & Squeezing ---
    print("\n--- 5. Quantum Lift & Squeezing ---")
    ql = QuantumLift(n_params=min(n_params, 100), fock_dim=16, hbar=0.1)

    # Kohärenter Zustand für einen Parameter
    alpha = torch.tensor(1.5)
    coh = ql.coherent_state(alpha)
    n_mean = ql.number_operator_expectation(coh)
    print(f"  Kohärenter Zustand |α=1.5⟩: ⟨n⟩ = {n_mean:.2f} (erwartet: {1.5**2:.2f})")

    # Squeezing
    small_params = torch.cat([p.flatten() for p in model.parameters()])[:100]
    compressed, sq_stats = ql.compress_via_squeezing(small_params, squeeze_param=0.5)
    print(f"  Squeezing (r=0.5):")
    print(f"    Bias: {sq_stats['bias']:.4f}")
    print(f"    Unsicherheit: {sq_stats['mean_uncertainty']:.4f}")
    print(f"    Qualität: {sq_stats['compression_quality']:.4f}")

    # Fidelity zwischen Original und Squeezed
    state_orig = ql.coherent_state(torch.tensor(2.0))
    state_sq = ql.squeeze_operator(0.5) @ state_orig
    fid = ql.fidelity(state_orig, state_sq)
    print(f"  Fidelity (Original vs Squeezed): {fid:.4f}")


def demo_full_pipeline():
    """Demonstriere die vollständige KIMI-Q Pipeline."""
    from kimi_q.integration.pytorch_kimiq import KIMIQTrainer, KIMIQConfig

    print("\n" + "=" * 60)
    print("KIMI-Q: Vollständige Pipeline Demo")
    print("=" * 60)

    # Datensatz
    X_train, y_train = create_toy_dataset(800)
    X_val, y_val = create_toy_dataset(200)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Modell
    model = ToyNet()
    loss_fn = nn.CrossEntropyLoss()

    # KIMI-Q Konfiguration
    config = KIMIQConfig(
        hbar=0.1,
        alpha=0.01,
        beta=0.001,
        dt=0.01,
        learning_rate=0.01,
        ricci_flow_interval=20,     # Ricci-Schritt alle 20 Batches
        target_sparsity=0.3,
        bulk_dimension=3,
        fock_dim=16,
        squeeze_param=0.5,
    )

    # Trainer
    trainer = KIMIQTrainer(model, config)

    # Phase 1: Training
    print("\n--- Phase 1: KIMI-Q Training ---")
    history = trainer.train(
        train_loader, loss_fn,
        epochs=5,
        val_dataloader=val_loader,
    )

    # Phase 2: Analyse
    print("\n--- Phase 2: Geometrische Analyse ---")
    analysis = trainer.analyze()

    # Phase 3: Kompression
    print("\n--- Phase 3: Holographische Kompression ---")

    # Evaluiere vor Kompression
    pre_stats = trainer.evaluate(val_loader, loss_fn)
    print(f"\nVor Kompression: Loss={pre_stats['loss']:.4f}, Acc={pre_stats['accuracy']:.4f}")

    # Komprimiere
    comp_stats = trainer.compress(method="full", target_sparsity=0.3)

    # Evaluiere nach Kompression
    post_stats = trainer.evaluate(val_loader, loss_fn)
    print(f"Nach Kompression: Loss={post_stats['loss']:.4f}, Acc={post_stats['accuracy']:.4f}")

    accuracy_drop = pre_stats["accuracy"] - post_stats["accuracy"]
    print(f"Genauigkeitsverlust: {accuracy_drop:.4f}")

    # Phase 4: Revival
    print("\n--- Phase 4: Zero-Shot Revival ---")
    revival_stats = trainer.revival()

    post_revival = trainer.evaluate(val_loader, loss_fn)
    print(f"Nach Revival: Loss={post_revival['loss']:.4f}, Acc={post_revival['accuracy']:.4f}")

    # Zusammenfassung
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameter: {n_params}")
    print(f"  Training: {len(history.losses)} Epochen")
    print(f"  Finale Genauigkeit: {history.accuracies[-1]:.4f}")
    if "total_compression" in comp_stats:
        print(f"  Kompression: {comp_stats['total_compression']:.1f}x")
    print(f"  Genauigkeit vor Kompression: {pre_stats['accuracy']:.4f}")
    print(f"  Genauigkeit nach Kompression: {post_stats['accuracy']:.4f}")
    print(f"  Genauigkeit nach Revival: {post_revival['accuracy']:.4f}")
    print(f"  Bell-Paare im Bulk: {len(trainer.pruner.bell_pairs)}")


if __name__ == "__main__":
    print("KIMI-Q: Kähler Information Manifold Inverse-Quantum")
    print("Das erste holographische Neural-Collapse-System")
    print()

    demo_individual_components()
    demo_full_pipeline()

    print("\n\nDemo abgeschlossen.")
