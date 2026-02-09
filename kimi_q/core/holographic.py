"""
Holographische Entropiegrenzen (Ryu-Takayanagi-Analogon).

Die Kompressionsrate wird durch die Ryu-Takayanagi-Formel bestimmt:

    S_NN = Area(γ_minimal) / (4 G_N)

wobei γ_minimal die minimale Fläche im Bulk-AdS-Raum ist, die der
Verschränkungsentropie der Netzwerkschicht entspricht.

Das holographische Prinzip impliziert:
    Effektive Freiheitsgrade ~ O(n^{(d-1)/d})

Für d=3: O(n^{2/3}) statt O(n).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List


class HolographicEntropy:
    """
    Holographische Entropiegrenzen für neuronale Netzkompression.

    Implementiert das Ryu-Takayanagi-Analogon für Deep Learning:
    Die minimale Verschränkungsfläche im Bulk-Parameterraum bestimmt
    die theoretische Kompressionsgrenze.
    """

    def __init__(
        self,
        bulk_dimension: int = 3,
        newton_constant: float = 1.0,
        ads_radius: float = 1.0,
    ):
        """
        Args:
            bulk_dimension: Dimension des Bulk-AdS-Raums (d+1)
            newton_constant: G_N (Lernrate-Äquivalent)
            ads_radius: Radius des Anti-de-Sitter-Raums
        """
        self.d = bulk_dimension
        self.G_N = newton_constant
        self.L = ads_radius

    def ryu_takayanagi_entropy(
        self,
        model: nn.Module,
        layer_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Berechne die holographische Entropie via Ryu-Takayanagi.

        S_NN = Area(γ_minimal) / (4 G_N)

        Die minimale Fläche wird aus der Singulärwertzerlegung der
        Gewichtsmatrix approximiert: Die Singulärwerte bestimmen die
        "Fläche" der minimalen Hyperfläche im Bulk.

        Args:
            model: Neuronales Netz
            layer_name: Spezifischer Layer (Standard: alle)

        Returns:
            entropies: Dict mit Entropie pro Layer
        """
        results = {}

        for name, param in model.named_parameters():
            if layer_name is not None and name != layer_name:
                continue
            if param.dim() < 2:
                continue

            W = param.data

            # SVD: W = U Σ V†
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            # Minimale Fläche ≈ Summe der Singulärwerte (Nuklear-Norm)
            # Die "Fläche" der RT-Hyperfläche ist proportional zur Nuklear-Norm
            area = S.sum().item()

            # Ryu-Takayanagi-Entropie
            S_RT = area / (4.0 * self.G_N)

            # Effektiver Rang (Anzahl signifikanter Singulärwerte)
            S_normalized = S / S.sum().clamp(min=1e-15)
            spectral_entropy = -(S_normalized * S_normalized.clamp(min=1e-15).log()).sum()
            effective_rank = spectral_entropy.exp().item()

            results[name] = {
                "ryu_takayanagi_entropy": S_RT,
                "minimal_area": area,
                "effective_rank": effective_rank,
                "full_rank": min(W.shape),
                "rank_ratio": effective_rank / min(W.shape),
            }

        return results

    def holographic_compression_bound(
        self,
        n_params: int,
    ) -> Dict[str, float]:
        """
        Berechne die holographische Kompressionsgrenze.

        Das holographische Prinzip impliziert:
            n_eff ~ O(n^{(d-1)/d})

        Für d=3 (Bulk-Dimension):
            n_eff ~ O(n^{2/3})

        Args:
            n_params: Anzahl der Parameter

        Returns:
            bounds: Kompressionsgrenzen
        """
        d = self.d
        # Holographische Skalierung
        n_eff = n_params ** ((d - 1) / d)
        max_compression = n_params / n_eff

        # Bekenstein-Grenze (maximale Information pro Fläche)
        bekenstein_bound = 2 * np.pi * self.L * n_params / (self.G_N * np.log(2))

        return {
            "original_params": n_params,
            "effective_params": n_eff,
            "max_compression_ratio": max_compression,
            "bulk_dimension": d,
            "bekenstein_bound_bits": bekenstein_bound,
            "holographic_scaling_exponent": (d - 1) / d,
        }

    def bulk_depth_from_compression(
        self,
        compression_ratio: float,
    ) -> float:
        """
        Berechne die Bulk-Tiefe aus der Kompressionsrate.

        Je mehr komprimiert wird, desto tiefer im AdS-Bulk:
            z_depth = L · log(C)

        wobei z die radiale AdS-Koordinate und C die Kompressionsrate ist.

        Args:
            compression_ratio: Kompressionsrate C = n/n_compressed

        Returns:
            z_depth: Tiefe im AdS-Bulk
        """
        return self.L * np.log(max(compression_ratio, 1.0))

    def compute_minimal_surface(
        self,
        weight_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Berechne die minimale Hyperfläche im Bulk.

        Die RT-Formel erfordert die Fläche der minimalen Hyperfläche,
        die homolog zur Boundary-Region ist. Für Gewichtsmatrizen
        approximieren wir dies durch die optimal abgeschnittene SVD.

        Args:
            weight_matrix: Gewichtsmatrix W (m × n)

        Returns:
            (gamma_minimal, area): Minimale Fläche und ihr Flächeninhalt
        """
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)

        # Minimale Fläche: Finde den optimalen SVD-Cutoff
        # Der Cutoff minimiert Area/G_N + Bulk-Beitrag
        total_sv = S.sum().item()
        n_sv = len(S)

        best_cutoff = n_sv
        best_cost = float('inf')

        for k in range(1, n_sv + 1):
            # Fläche der minimalen Hyperfläche (Top-k Singulärwerte)
            area_k = S[:k].sum().item()
            # Bulk-Beitrag (Residual-Energie)
            residual = S[k:].sum().item() if k < n_sv else 0.0
            # Gesamtkosten: RT-Entropie + Reconstruction Error
            cost = area_k / (4 * self.G_N) + residual ** 2
            if cost < best_cost:
                best_cost = cost
                best_cutoff = k

        # Minimale Fläche: Rekonstruktion mit optimalem Rang
        gamma = U[:, :best_cutoff] @ torch.diag(S[:best_cutoff]) @ Vh[:best_cutoff, :]
        area = S[:best_cutoff].sum().item()

        return gamma, area

    def holographic_projection(
        self,
        model: nn.Module,
        target_ratio: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Holographische Projektion: Projiziere das Netzwerk auf die
        effektiven Freiheitsgrade im Bulk.

        Args:
            model: Neuronales Netz
            target_ratio: Ziel-Kompressionsrate (Standard: holographisches Limit)

        Returns:
            stats: Projektionsstatistiken
        """
        total_params = sum(p.numel() for p in model.parameters())

        if target_ratio is None:
            bounds = self.holographic_compression_bound(total_params)
            target_ratio = bounds["max_compression_ratio"]

        total_original = 0
        total_compressed = 0

        for name, param in model.named_parameters():
            if param.dim() < 2:
                total_original += param.numel()
                total_compressed += param.numel()
                continue

            W = param.data
            m, n = W.shape
            total_original += m * n

            # Optimale Rang-Reduktion via minimale Fläche
            gamma, area = self.compute_minimal_surface(W)

            # Berechne Rang der Projektion
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            target_rank = max(1, int(min(m, n) / target_ratio))
            target_rank = min(target_rank, min(m, n))

            # Projiziere
            W_proj = U[:, :target_rank] @ torch.diag(S[:target_rank]) @ Vh[:target_rank, :]
            param.data.copy_(W_proj)

            total_compressed += target_rank * (m + n)  # Low-Rank-Speicher

        actual_ratio = total_original / max(total_compressed, 1)

        return {
            "original_params": total_original,
            "compressed_params": total_compressed,
            "compression_ratio": actual_ratio,
            "target_ratio": target_ratio,
            "bulk_depth": self.bulk_depth_from_compression(actual_ratio),
            "holographic_limit": self.holographic_compression_bound(total_params)["max_compression_ratio"],
        }

    def entanglement_wedge_reconstruction(
        self,
        model: nn.Module,
        layer_indices: List[int],
    ) -> torch.Tensor:
        """
        Entanglement-Wedge-Rekonstruktion: Rekonstruiere Bulk-Operatoren
        aus Boundary-Daten.

        In der AdS/CFT-Korrespondenz kann ein Bulk-Operator innerhalb
        des Entanglement Wedge einer Boundary-Region rekonstruiert werden.
        Hier rekonstruieren wir "tiefe" Features aus "flachen" Layern.

        Args:
            model: Neuronales Netz
            layer_indices: Indizes der Boundary-Layer

        Returns:
            bulk_operator: Rekonstruierter Bulk-Operator
        """
        params = list(model.parameters())
        boundary_weights = []

        for idx in layer_indices:
            if idx < len(params):
                boundary_weights.append(params[idx].data.flatten())

        if not boundary_weights:
            return torch.tensor([0.0])

        # Konkateniere Boundary-Daten
        boundary = torch.cat(boundary_weights)

        # Bulk-Rekonstruktion via HKLL-Integral (Kernel)
        # Smearing-Funktion: K(z, x) ~ z^Δ für konformes Gewicht Δ
        n = boundary.numel()
        z_values = torch.linspace(0.1, 1.0, n)  # Bulk-Radialkoordinate

        # Konformes Gewicht Δ aus der Dimension des Operators
        delta = (self.d - 1) / 2.0

        # HKLL-Kernel
        kernel = z_values ** delta

        # Bulk-Operator = ∫ K(z,x) O(x) dx
        bulk_op = boundary * kernel

        return bulk_op
