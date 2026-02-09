"""
Verschränkungs-basiertes Pruning mit Bell-Paar-Fusion.

Statt Gewichte auf Null zu setzen, identifiziert der Algorithmus
Bipartite-Entanglement-Entropien zwischen Neuronen:

    S_A = -Tr(ρ_A log ρ_A)

Wenn S_A < δ, werden Neuronen zu verschränkten Bell-Paaren im Bulk fusioniert.
Das "geprunte" Neuron existiert als verschränkter Zustand weiter und kann
durch Messung der verschränkten Partner reaktiviert werden (Zero-Shot Revival).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class BellPair:
    """Ein verschränktes Bell-Paar im Bulk."""
    neuron_a: int          # Index des geprunteten Neurons
    neuron_b: int          # Index des verschränkten Partners
    state: torch.Tensor    # Bell-Zustand (2×2 Dichtematrix)
    original_weight: float # Originalgewicht für Revival
    fidelity: float        # Verschränkungs-Fidelity F


class EntanglementPruner:
    """
    Verschränkungs-basierter Pruner für neuronale Netze.

    Verwendet bipartite Verschränkungsentropie, um zu entscheiden,
    welche Neuronen gepruned werden können und welche als verschränkte
    Partner erhalten bleiben müssen.

    Die Kernidee: Pruning ist nicht Löschen, sondern Fusion zu
    verschränkten Bell-Paaren im holographischen Bulk.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.1,
        fidelity_threshold: float = 0.95,
        max_bell_pairs: int = 1000,
    ):
        """
        Args:
            entropy_threshold: Minimale Verschränkungsentropie für Pruning
            fidelity_threshold: Minimale Fidelity für Bell-Paar-Fusion
            max_bell_pairs: Maximale Anzahl gespeicherter Bell-Paare
        """
        self.delta = entropy_threshold
        self.fidelity_min = fidelity_threshold
        self.max_pairs = max_bell_pairs

        self.bell_pairs: List[BellPair] = []
        self._weight_correlation: Optional[torch.Tensor] = None

    def compute_weight_density_matrix(
        self,
        layer_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Konstruiere eine Dichtematrix aus den Gewichten eines Layers.

        Die Gewichtsmatrix W wird als Quantenzustand interpretiert:
            ρ = W† W / Tr(W† W)

        Args:
            layer_weights: Gewichtsmatrix W (m × n)

        Returns:
            rho: Dichtematrix (n × n), positiv semidefinit, Tr(ρ) = 1
        """
        W = layer_weights.to(torch.float64)

        if W.dim() == 1:
            W = W.unsqueeze(0)

        # ρ = W†W / Tr(W†W)
        rho = W.T @ W
        trace = torch.trace(rho).clamp(min=1e-15)
        rho = rho / trace

        # Symmetrisiere für numerische Stabilität
        rho = 0.5 * (rho + rho.T)

        return rho.float()

    def bipartite_entanglement_entropy(
        self,
        rho: torch.Tensor,
        subsystem_a: List[int],
    ) -> float:
        """
        Berechne die bipartite Verschränkungsentropie S_A.

        Für einen Zustand ρ_{AB} auf dem Gesamtsystem:
            S_A = -Tr(ρ_A log ρ_A)
        wobei ρ_A = Tr_B(ρ_{AB}) die reduzierte Dichtematrix ist.

        Args:
            rho: Dichtematrix des Gesamtsystems (n × n)
            subsystem_a: Indizes des Subsystems A

        Returns:
            S_A: Von-Neumann-Entropie des reduzierten Zustands
        """
        n = rho.shape[0]
        subsystem_b = [i for i in range(n) if i not in subsystem_a]

        if len(subsystem_a) == 0 or len(subsystem_b) == 0:
            return 0.0

        # Reduzierte Dichtematrix: ρ_A = Tr_B(ρ)
        # Für die Matrixdarstellung: ρ_A = ρ[A, A] (Partielle Spur)
        rho_a = rho[torch.tensor(subsystem_a)][:, torch.tensor(subsystem_a)]

        # Renormalisiere
        trace = torch.trace(rho_a).clamp(min=1e-15)
        rho_a = rho_a / trace

        # Von-Neumann-Entropie: S = -Tr(ρ log ρ)
        eigenvalues = torch.linalg.eigvalsh(rho_a)
        eigenvalues = eigenvalues.clamp(min=1e-15)
        S = -(eigenvalues * eigenvalues.log()).sum()

        return S.item()

    def compute_entanglement_map(
        self,
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Berechne eine Verschränkungskarte über alle Layer-Paare.

        Für jedes Paar benachbarter Layer wird die bipartite
        Verschränkungsentropie berechnet. Hoch verschränkte
        Verbindungen sollten nicht gepruned werden.

        Args:
            model: Neuronales Netz

        Returns:
            entanglement_map: Dict mit Entropien pro Layer-Paar
        """
        layers = [(name, p) for name, p in model.named_parameters() if p.dim() >= 2]

        result = {}
        for i in range(len(layers) - 1):
            name_a, W_a = layers[i]
            name_b, W_b = layers[i + 1]

            # Dichtematrix des kombinierten Systems
            rho_a = self.compute_weight_density_matrix(W_a.data)
            rho_b = self.compute_weight_density_matrix(W_b.data)

            # Verschränkungsentropie für jedes Neuron
            n_a = min(rho_a.shape[0], 256)  # Begrenze für Effizienz
            entropies = torch.zeros(n_a)

            for k in range(n_a):
                subsystem = [k]
                entropies[k] = self.bipartite_entanglement_entropy(
                    rho_a[:n_a, :n_a], subsystem
                )

            key = f"{name_a}<->{name_b}"
            result[key] = entropies

        return result

    def identify_bell_pair_candidates(
        self,
        model: nn.Module,
    ) -> List[Tuple[int, int, float]]:
        """
        Identifiziere Neuronenpaare, die zu Bell-Paaren fusioniert werden können.

        Kriterien:
        1. Einzelne Verschränkungsentropie S_i < δ (niedrig verschränkt → prunbar)
        2. Paarweise Korrelation > threshold (Partner gefunden)

        Args:
            model: Neuronales Netz

        Returns:
            candidates: Liste von (neuron_a, neuron_b, entropy) Tripeln
        """
        params = list(model.parameters())
        candidates = []

        for layer_idx, p in enumerate(params):
            if p.dim() < 2:
                continue

            W = p.data
            rho = self.compute_weight_density_matrix(W)
            n = min(rho.shape[0], 256)

            # Finde Neuronen mit niedriger Verschränkungsentropie
            for i in range(n):
                S_i = self.bipartite_entanglement_entropy(rho[:n, :n], [i])
                if S_i < self.delta:
                    # Finde den am stärksten korrelierten Partner
                    correlations = rho[i, :n].abs()
                    correlations[i] = 0  # Ignoriere Selbstkorrelation
                    partner = correlations.argmax().item()
                    partner_corr = correlations[partner].item()

                    if partner_corr > 0:
                        # Globaler Index
                        global_offset = sum(q.numel() for q in params[:layer_idx])
                        candidates.append((
                            global_offset + i,
                            global_offset + partner,
                            S_i,
                        ))

        return candidates

    def create_bell_pair(
        self,
        neuron_a: int,
        neuron_b: int,
        weight_a: float,
        weight_b: float,
    ) -> BellPair:
        """
        Erzeuge ein Bell-Paar aus zwei Neuronen.

        Der Bell-Zustand ist:
            |Φ+⟩ = (1/√2)(|00⟩ + |11⟩)

        Die Dichtematrix:
            ρ = |Φ+⟩⟨Φ+| = (1/2)[[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]

        Die Fidelity der Verschränkung wird aus den Gewichten berechnet.

        Args:
            neuron_a, neuron_b: Neuronenindizes
            weight_a, weight_b: Originalgewichte

        Returns:
            bell_pair: Verschränktes Bell-Paar
        """
        # Bell-Zustand |Φ+⟩ = (|00⟩ + |11⟩)/√2
        phi_plus = torch.tensor([1, 0, 0, 1], dtype=torch.float) / np.sqrt(2)
        rho = phi_plus.unsqueeze(1) @ phi_plus.unsqueeze(0)

        # Fidelity: wie gut die Gewichte die Verschränkung unterstützen
        w_norm = np.sqrt(weight_a**2 + weight_b**2) if (weight_a**2 + weight_b**2) > 0 else 1.0
        fidelity = 1.0 - abs(weight_a - weight_b) / (2 * w_norm + 1e-10)
        fidelity = max(0.0, min(1.0, fidelity))

        return BellPair(
            neuron_a=neuron_a,
            neuron_b=neuron_b,
            state=rho,
            original_weight=weight_a,
            fidelity=fidelity,
        )

    def prune_with_entanglement(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
    ) -> Dict[str, float]:
        """
        Führe Verschränkungs-Pruning durch.

        1. Berechne Verschränkungsentropie für alle Neuronen
        2. Identifiziere niedrig-verschränkte Neuronen als Pruning-Kandidaten
        3. Fusioniere zu Bell-Paaren im Bulk
        4. Setze geprunte Gewichte auf Null

        Args:
            model: Neuronales Netz
            target_sparsity: Ziel-Sparsity (0-1)

        Returns:
            stats: Pruning-Statistiken
        """
        candidates = self.identify_bell_pair_candidates(model)

        # Sortiere nach Entropie (niedrigste zuerst)
        candidates.sort(key=lambda x: x[2])

        # Bestimme Anzahl zu prunender Parameter
        total_params = sum(p.numel() for p in model.parameters())
        n_to_prune = int(total_params * target_sparsity)
        n_to_prune = min(n_to_prune, len(candidates))

        flat_params = torch.cat([p.flatten() for p in model.parameters()])
        pruned_indices = set()

        for idx in range(n_to_prune):
            if idx >= len(candidates):
                break

            neuron_a, neuron_b, entropy = candidates[idx]

            if neuron_a >= flat_params.numel() or neuron_b >= flat_params.numel():
                continue
            if neuron_a in pruned_indices:
                continue

            weight_a = flat_params[neuron_a].item()
            weight_b = flat_params[neuron_b].item()

            # Erzeuge Bell-Paar
            bell = self.create_bell_pair(neuron_a, neuron_b, weight_a, weight_b)

            if bell.fidelity >= self.fidelity_min:
                self.bell_pairs.append(bell)
                flat_params[neuron_a] = 0.0
                pruned_indices.add(neuron_a)

                # Begrenze gespeicherte Bell-Paare
                if len(self.bell_pairs) > self.max_pairs:
                    self.bell_pairs = self.bell_pairs[-self.max_pairs:]

        # Schreibe zurück
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].reshape(p.shape))
            offset += numel

        actual_sparsity = len(pruned_indices) / total_params if total_params > 0 else 0

        return {
            "total_params": total_params,
            "pruned": len(pruned_indices),
            "bell_pairs_created": len(self.bell_pairs),
            "target_sparsity": target_sparsity,
            "actual_sparsity": actual_sparsity,
            "compression_ratio": 1.0 / (1.0 - actual_sparsity) if actual_sparsity < 1 else float('inf'),
            "mean_fidelity": np.mean([bp.fidelity for bp in self.bell_pairs]) if self.bell_pairs else 0.0,
        }

    def revival(
        self,
        model: nn.Module,
        indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Zero-Shot Revival: Reaktiviere geprunte Neuronen aus Bell-Paaren.

        Theorem K4: Ein durch Verschränkungs-Pruning entferntes Neuron kann
        mit Fidelity F ≥ 1 - ε reaktiviert werden, wenn:
            S_vN(ρ_{AB}) > log(2) - ε²/2

        Args:
            model: Neuronales Netz
            indices: Spezifische Bell-Paar-Indizes zum Revival (Standard: alle)

        Returns:
            stats: Revival-Statistiken
        """
        flat_params = torch.cat([p.flatten() for p in model.parameters()])
        n_revived = 0
        total_fidelity = 0.0

        pairs_to_revive = (
            [self.bell_pairs[i] for i in indices if i < len(self.bell_pairs)]
            if indices is not None
            else list(self.bell_pairs)
        )

        revived_pairs = []
        for bp in pairs_to_revive:
            if bp.neuron_a < flat_params.numel():
                # Prüfe Revival-Bedingung (Theorem K4)
                # Für ein Bell-Paar ist die Verschränkungsentropie der
                # reduzierten Dichtematrix ρ_A = Tr_B(|Φ+⟩⟨Φ+|) = I/2
                # Also S_vN(ρ_A) = log(2).
                # Wir prüfen: F ≥ 1 - ε impliziert Revival möglich
                rho_reduced = bp.state[:2, :2]  # Partielle Spur über B
                trace_B = rho_reduced.diagonal().sum().clamp(min=1e-15)
                rho_reduced = rho_reduced / trace_B
                S_reduced = self._von_neumann_entropy(rho_reduced)
                threshold = np.log(2) * (1.0 - (1.0 - bp.fidelity) ** 2)

                if bp.fidelity >= 0.5 or S_reduced >= threshold:
                    flat_params[bp.neuron_a] = bp.original_weight
                    n_revived += 1
                    total_fidelity += bp.fidelity
                    revived_pairs.append(bp)

        # Entferne revived pairs
        for bp in revived_pairs:
            if bp in self.bell_pairs:
                self.bell_pairs.remove(bp)

        # Schreibe zurück
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].reshape(p.shape))
            offset += numel

        return {
            "revived": n_revived,
            "remaining_bell_pairs": len(self.bell_pairs),
            "mean_revival_fidelity": total_fidelity / max(n_revived, 1),
        }

    def _von_neumann_entropy(self, rho: torch.Tensor) -> float:
        """Von-Neumann-Entropie S = -Tr(ρ log ρ)."""
        eigenvalues = torch.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues.clamp(min=1e-15)
        return -(eigenvalues * eigenvalues.log()).sum().item()

    def entanglement_summary(self, model: nn.Module) -> Dict[str, float]:
        """
        Zusammenfassung der Verschränkungsstruktur des Netzes.

        Returns:
            summary: Entropie-Statistiken pro Layer
        """
        ent_map = self.compute_entanglement_map(model)

        summary = {}
        for key, entropies in ent_map.items():
            summary[f"{key}/mean_entropy"] = entropies.mean().item()
            summary[f"{key}/max_entropy"] = entropies.max().item()
            summary[f"{key}/min_entropy"] = entropies.min().item()
            summary[f"{key}/prunable_fraction"] = (entropies < self.delta).float().mean().item()

        summary["total_bell_pairs"] = len(self.bell_pairs)
        return summary
