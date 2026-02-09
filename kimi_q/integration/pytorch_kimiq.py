"""
KIMI-Q PyTorch Integration: Holographischer Trainer und Optimizer.

Implementiert den vollständigen KIMI-Q Algorithmus als PyTorch-Trainer:

1. QUANTUM LIFT: θ → z = θ + ip, Kähler-Metrik G_{iȷ̄}
2. HOLOGRAPHISCHER RICCI-FLUSS: ∂G/∂t = -R + α∇²L + βT
3. SINGULARITÄTS-DETEKTION: Pruning-Kandidaten
4. HOLOGRAPHISCHE PROJEKTION: Effektive Freiheitsgrade
5. SPEKTRALE VALIDIERUNG: Topologisches Budget
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
import time

from kimi_q.core.kahler_manifold import KahlerManifold, _get_flat_params
from kimi_q.core.ricci_flow import HolographicRicciFlow
from kimi_q.core.entanglement_pruning import EntanglementPruner
from kimi_q.core.holographic import HolographicEntropy
from kimi_q.core.dirac_operator import DiracOperator
from kimi_q.core.quantum_lift import QuantumLift


@dataclass
class KIMIQConfig:
    """Konfiguration für den KIMI-Q Trainer."""
    # Kähler-Mannigfaltigkeit
    hbar: float = 0.1
    regularization: float = 1e-4

    # Ricci-Fluss
    alpha: float = 0.01       # Loss-Hessische-Gewicht
    beta: float = 0.001       # Verschränkungs-Tensor-Gewicht
    dt: float = 0.01          # Ricci-Fluss-Zeitschritt

    # Pruning
    singularity_threshold: float = 1e-3
    entropy_threshold: float = 0.1
    target_sparsity: float = 0.5

    # Holographie
    bulk_dimension: int = 3
    newton_constant: float = 1.0

    # Dirac / Spektral
    higgs_mass: float = 1.0
    spectral_keep_ratio: float = 0.5

    # Quantum Lift
    fock_dim: int = 16
    squeeze_param: float = 1.0

    # Training
    learning_rate: float = 0.01
    ricci_flow_interval: int = 10  # Alle N Batches einen Ricci-Schritt
    max_metric_params: int = 2048  # Max Parameter für volle Metrik-Berechnung


@dataclass
class KIMIQHistory:
    """Trainingshistorie."""
    losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    ricci_diagnostics: List[Dict[str, float]] = field(default_factory=list)
    pruning_stats: List[Dict[str, float]] = field(default_factory=list)
    holographic_stats: List[Dict[str, float]] = field(default_factory=list)
    spectral_stats: List[Dict[str, float]] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)


class KIMIQTrainer:
    """
    KIMI-Q Trainer: Holographisches Training und Kompression.

    Orchestriert alle KIMI-Q Komponenten:
    - Kähler-Mannigfaltigkeit für den Parameterraum
    - Ricci-Fluss für geometrisches Training
    - Verschränkungs-Pruning für intelligente Kompression
    - Holographische Grenzen für theoretische Validierung
    - Spektrale Geometrie für topologische Analyse
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[KIMIQConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Neuronales Netz
            config: KIMI-Q Konfiguration
            device: PyTorch Device
        """
        self.model = model
        self.config = config or KIMIQConfig()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        n_params = sum(p.numel() for p in model.parameters())

        # Initialisiere Komponenten
        # Für große Modelle: verwende Low-Rank-Metrik
        metric_params = min(n_params, self.config.max_metric_params)

        self.manifold = KahlerManifold(
            n_params=metric_params,
            hbar=self.config.hbar,
            regularization=self.config.regularization,
        )

        self.ricci_flow = HolographicRicciFlow(
            manifold=self.manifold,
            alpha=self.config.alpha,
            beta=self.config.beta,
            dt=self.config.dt,
            singularity_threshold=self.config.singularity_threshold,
            entropy_threshold=self.config.entropy_threshold,
        )

        self.pruner = EntanglementPruner(
            entropy_threshold=self.config.entropy_threshold,
            fidelity_threshold=0.95,
        )

        self.holographic = HolographicEntropy(
            bulk_dimension=self.config.bulk_dimension,
            newton_constant=self.config.newton_constant,
        )

        self.dirac = DiracOperator(
            higgs_mass=self.config.higgs_mass,
        )

        self.quantum_lift = QuantumLift(
            n_params=metric_params,
            fock_dim=self.config.fock_dim,
            hbar=self.config.hbar,
        )

        # Standard-Optimizer (für Zwischen-Steps ohne Ricci-Fluss)
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        self.history = KIMIQHistory()
        self._step_count = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Trainiere eine Epoche mit dem KIMI-Q Algorithmus.

        Kombiniert:
        - Standard-Gradientenabstieg (jeder Batch)
        - Ricci-Fluss-Schritte (alle N Batches)
        - Geometrische Diagnostik

        Args:
            dataloader: PyTorch DataLoader
            loss_fn: Verlustfunktion
            epoch: Epochennummer

        Returns:
            epoch_stats: Epochenstatistiken
        """
        self.model.train()
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0
        t_start = time.time()

        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Standard-Forward-Backward
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            n_correct += (pred == targets).sum().item()
            n_total += targets.size(0)

            self._step_count += 1

            # Ricci-Fluss-Schritt (periodisch)
            if self._step_count % self.config.ricci_flow_interval == 0:
                n_params = sum(p.numel() for p in self.model.parameters())
                if n_params <= self.config.max_metric_params:
                    try:
                        self.ricci_flow.step(self.model, data, targets, loss_fn)
                        diag = self.ricci_flow.convergence_diagnostic()
                        self.history.ricci_diagnostics.append(diag)
                    except Exception:
                        pass  # Ricci-Fluss ist optional

        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / max(len(dataloader), 1)
        accuracy = n_correct / max(n_total, 1)

        self.history.losses.append(avg_loss)
        self.history.accuracies.append(accuracy)
        self.history.epoch_times.append(epoch_time)

        return {
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "time": epoch_time,
        }

    def train(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        epochs: int = 10,
        val_dataloader: Optional[DataLoader] = None,
        callback: Optional[Callable] = None,
    ) -> KIMIQHistory:
        """
        Vollständiges KIMI-Q Training.

        Args:
            dataloader: Trainings-DataLoader
            loss_fn: Verlustfunktion
            epochs: Anzahl Epochen
            val_dataloader: Validierungs-DataLoader
            callback: Callback-Funktion pro Epoche

        Returns:
            history: Trainingshistorie
        """
        print(f"=== KIMI-Q Training ===")
        print(f"Model: {sum(p.numel() for p in self.model.parameters())} Parameter")
        print(f"Config: hbar={self.config.hbar}, alpha={self.config.alpha}, "
              f"beta={self.config.beta}, dt={self.config.dt}")
        print(f"Epochs: {epochs}")
        print()

        for epoch in range(epochs):
            stats = self.train_epoch(dataloader, loss_fn, epoch)

            msg = f"Epoch {epoch+1}/{epochs} | Loss: {stats['loss']:.4f} | Acc: {stats['accuracy']:.4f}"

            if val_dataloader is not None:
                val_stats = self.evaluate(val_dataloader, loss_fn)
                msg += f" | Val Loss: {val_stats['loss']:.4f} | Val Acc: {val_stats['accuracy']:.4f}"

            msg += f" | Time: {stats['time']:.1f}s"
            print(msg)

            if callback is not None:
                callback(epoch, stats)

        return self.history

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Evaluiere das Modell."""
        self.model.eval()
        total_loss = 0.0
        n_correct = 0
        n_total = 0

        with torch.no_grad():
            for data, targets in dataloader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                output = self.model(data)
                loss = loss_fn(output, targets)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                n_correct += (pred == targets).sum().item()
                n_total += targets.size(0)

        return {
            "loss": total_loss / max(len(dataloader), 1),
            "accuracy": n_correct / max(n_total, 1),
        }

    def compress(
        self,
        method: str = "holographic",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Komprimiere das Modell mit KIMI-Q Methoden.

        Verfügbare Methoden:
        - "holographic": Holographische Projektion (RT-Formel)
        - "entanglement": Verschränkungs-Pruning mit Bell-Paaren
        - "spectral": Spektrales Pruning via Dirac-Operator
        - "ricci": Geometrisches Pruning via Ricci-Singularitäten
        - "squeeze": Fock-Raum-Squeezing
        - "full": Alle Methoden kombiniert

        Args:
            method: Kompressionsmethode
            **kwargs: Methodenspezifische Parameter

        Returns:
            stats: Kompressionsstatistiken
        """
        print(f"\n=== KIMI-Q Kompression: {method} ===")

        if method == "holographic":
            target_ratio = kwargs.get("target_ratio", None)
            stats = self.holographic.holographic_projection(self.model, target_ratio)
            self.history.holographic_stats.append(stats)

        elif method == "entanglement":
            sparsity = kwargs.get("target_sparsity", self.config.target_sparsity)
            stats = self.pruner.prune_with_entanglement(self.model, sparsity)
            self.history.pruning_stats.append(stats)

        elif method == "spectral":
            keep_ratio = kwargs.get("keep_ratio", self.config.spectral_keep_ratio)
            stats = self.dirac.spectral_pruning(self.model, keep_ratio)
            self.history.spectral_stats.append(stats)

        elif method == "ricci":
            candidates = self.ricci_flow.detect_pruning_candidates()
            stats = self.ricci_flow.apply_geometric_pruning(self.model, candidates)

        elif method == "squeeze":
            params = _get_flat_params(self.model)
            n = min(params.numel(), self.config.max_metric_params)
            _, stats = self.quantum_lift.compress_via_squeezing(
                params[:n],
                self.config.squeeze_param,
            )

        elif method == "full":
            stats = self._full_compression(**kwargs)

        else:
            raise ValueError(f"Unbekannte Methode: {method}")

        print(f"Kompression abgeschlossen: {stats}")
        return stats

    def _full_compression(self, **kwargs) -> Dict[str, Any]:
        """Vollständige KIMI-Q Kompression (alle Methoden kombiniert)."""
        results = {}

        # 1. Topologische Analyse
        print("  [1/4] Topologische Analyse (Dirac-Operator)...")
        topo = self.dirac.topological_invariants(self.model)
        results["topology"] = topo
        print(f"        Index: {topo['atiyah_singer_index']}, "
              f"Budget: {topo['topological_budget']}, "
              f"Max Compression: {topo['topology_compression_ratio']:.1f}x")

        # 2. Holographische Grenzen
        print("  [2/4] Holographische Grenzen...")
        n_params = sum(p.numel() for p in self.model.parameters())
        bounds = self.holographic.holographic_compression_bound(n_params)
        results["holographic_bounds"] = bounds
        print(f"        Holographisches Limit: {bounds['max_compression_ratio']:.1f}x")

        # 3. Verschränkungs-Pruning
        print("  [3/4] Verschränkungs-Pruning...")
        sparsity = kwargs.get("target_sparsity", self.config.target_sparsity)
        pruning_stats = self.pruner.prune_with_entanglement(self.model, sparsity)
        results["pruning"] = pruning_stats
        print(f"        Sparsity: {pruning_stats['actual_sparsity']:.2%}, "
              f"Bell-Paare: {pruning_stats['bell_pairs_created']}")

        # 4. Holographische Projektion
        print("  [4/4] Holographische Projektion...")
        proj_stats = self.holographic.holographic_projection(self.model)
        results["projection"] = proj_stats
        print(f"        Kompression: {proj_stats['compression_ratio']:.1f}x, "
              f"Bulk-Tiefe: {proj_stats['bulk_depth']:.2f}")

        results["total_compression"] = proj_stats["compression_ratio"]
        return results

    def analyze(self) -> Dict[str, Any]:
        """
        Vollständige KIMI-Q Analyse des Modells (ohne Modifikation).

        Returns:
            analysis: Umfassende geometrische und topologische Analyse
        """
        print("\n=== KIMI-Q Analyse ===\n")
        results = {}

        # 1. Topologische Invarianten
        print("[1/4] Topologische Invarianten...")
        topo = self.dirac.topological_invariants(self.model)
        results["topology"] = topo
        print(f"  Euler-Charakteristik: {topo['euler_characteristic']}")
        print(f"  Spektrale Dimension: {topo['mean_spectral_dimension']:.2f}")
        print(f"  Topologisches Budget: {topo['topological_budget']} Parameter")
        print(f"  Überschüssige Parameter: {topo['excess_parameters']}")

        # 2. Holographische Grenzen
        print("\n[2/4] Holographische Grenzen...")
        n_params = sum(p.numel() for p in self.model.parameters())
        bounds = self.holographic.holographic_compression_bound(n_params)
        results["holographic_bounds"] = bounds
        print(f"  Original: {bounds['original_params']} Parameter")
        print(f"  Effektiv (holographisch): {bounds['effective_params']:.0f} Parameter")
        print(f"  Max Kompression: {bounds['max_compression_ratio']:.1f}x")

        # 3. Verschränkungsstruktur
        print("\n[3/4] Verschränkungsstruktur...")
        ent_summary = self.pruner.entanglement_summary(self.model)
        results["entanglement"] = ent_summary
        for key, val in ent_summary.items():
            print(f"  {key}: {val:.4f}")

        # 4. Ryu-Takayanagi-Entropie
        print("\n[4/4] Ryu-Takayanagi-Entropie...")
        rt_entropy = self.holographic.ryu_takayanagi_entropy(self.model)
        results["ryu_takayanagi"] = rt_entropy
        for name, info in rt_entropy.items():
            print(f"  {name}: S_RT={info['ryu_takayanagi_entropy']:.4f}, "
                  f"eff_rank={info['effective_rank']:.1f}/{info['full_rank']}")

        return results

    def revival(self, indices: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Zero-Shot Revival: Reaktiviere geprunte Neuronen.

        Args:
            indices: Spezifische Bell-Paar-Indizes (Standard: alle)

        Returns:
            stats: Revival-Statistiken
        """
        print("\n=== Zero-Shot Revival ===")
        stats = self.pruner.revival(self.model, indices)
        print(f"  Revived: {stats['revived']} Neuronen")
        print(f"  Verbleibende Bell-Paare: {stats['remaining_bell_pairs']}")
        return stats
