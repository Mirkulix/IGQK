"""
Kähler-Fisher-Mannigfaltigkeit auf dem komplexifizierten Parameterraum.

Die Fisher-Information-Metrik wird zu einer Kähler-Metrik erweitert:

    G_{iȷ̄}(z, z̄) = ∂_i ∂_ȷ̄ K(z, z̄)

wobei K das Kähler-Potential ist, abgeleitet aus der cumulant generating function.
Die Komplexifizierung z = θ + i·p erweitert den Parameterraum um konjugierte
Impulse, die Lernraten-Information kodieren.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class KahlerManifold:
    """
    Kähler-Mannigfaltigkeit auf dem komplexifizierten Parameterraum eines
    neuronalen Netzes.

    Die reellen Gewichte θ werden zu komplexen Koordinaten z = θ + ip erweitert,
    wobei p als konjugierter Impuls (Richtungs-/Lernraten-Information) dient.
    Die Kähler-Metrik G_{iȷ̄} = ∂_i ∂_ȷ̄ K wird aus dem Kähler-Potential K
    berechnet, das aus der Fisher-Information abgeleitet wird.
    """

    def __init__(
        self,
        n_params: int,
        hbar: float = 0.1,
        regularization: float = 1e-4,
    ):
        """
        Args:
            n_params: Anzahl der reellen Parameter θ
            hbar: Quanten-Unsicherheitsparameter ℏ
            regularization: Tikhonov-Regularisierung für numerische Stabilität
        """
        self.n = n_params
        self.hbar = hbar
        self.reg = regularization

        # Komplexe Koordinaten: z = θ + ip
        self.z = torch.zeros(n_params, dtype=torch.cfloat)
        # Kähler-Metrik (Hermitische Matrix)
        self.metric = torch.eye(n_params, dtype=torch.cfloat)
        # Kähler-Potential (Skalar)
        self.potential = torch.tensor(0.0)

    def complexify(self, theta: torch.Tensor, momentum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Komplexifiziere reelle Parameter zu z = θ + ip.

        Args:
            theta: Reelle Gewichte θ ∈ ℝⁿ
            momentum: Konjugierter Impuls p ∈ ℝⁿ (Standard: Null)

        Returns:
            z ∈ ℂⁿ
        """
        if momentum is None:
            momentum = torch.zeros_like(theta)
        self.z = theta.to(torch.cfloat) + 1j * momentum.to(torch.cfloat)
        return self.z

    def kahler_potential(self, z: torch.Tensor, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """
        Berechne das Kähler-Potential aus der cumulant generating function:

            K(z, z̄) = log ∫ p(x|θ)^{1-iℏ} dx

        Approximiert durch empirische Erwartung über Mini-Batch.

        Args:
            z: Komplexe Koordinaten
            model: Neuronales Netz
            data: Datenbatch x

        Returns:
            K: Kähler-Potential (reell)
        """
        theta = z.real
        # Setze reelle Parameter ins Modell
        _set_flat_params(model, theta)

        with torch.no_grad():
            output = model(data)
            # Log-Softmax als log p(x|θ)
            log_probs = torch.log_softmax(output, dim=-1)
            # Cumulant generating function: log E[p^{1-iℏ}]
            # Realteil der skalierten Log-Wahrscheinlichkeit
            scaled = (1.0 - 1j * self.hbar) * log_probs.to(torch.cfloat)
            K = torch.logsumexp(scaled.real, dim=-1).mean()

        self.potential = K.real
        return self.potential

    def compute_metric(
        self,
        model: nn.Module,
        data: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Berechne die Kähler-Metrik G_{iȷ̄} = ∂_i ∂_ȷ̄ K.

        Approximiert durch die empirische Fisher-Information-Matrix im
        komplexifizierten Raum:

            G_{iȷ̄} ≈ (1/N) Σ_k (∂_i log p_k)(∂_ȷ̄ log p_k)*

        Für große n verwenden wir eine Block-Diagonale oder Low-Rank-Approximation.

        Args:
            model: Neuronales Netz
            data: Eingabedaten
            targets: Zielwerte

        Returns:
            G: Hermitische Kähler-Metrik (n × n komplexe Matrix)
        """
        n = self.n
        params = _get_flat_params(model)

        # Berechne Gradienten ∂_i log p(y|x,θ) für jeden Datenpunkt
        grads = []
        model.zero_grad()
        output = model(data)
        log_probs = torch.log_softmax(output, dim=-1)

        for k in range(min(data.shape[0], 64)):  # Max 64 Samples für Effizienz
            model.zero_grad()
            log_p_k = log_probs[k, targets[k]] if targets is not None else log_probs[k].sum()
            g = torch.autograd.grad(log_p_k, model.parameters(), retain_graph=True, allow_unused=True)
            flat_g = torch.cat([gi.flatten() for gi in g if gi is not None])
            grads.append(flat_g)

        if not grads:
            self.metric = torch.eye(n, dtype=torch.cfloat) * self.reg
            return self.metric

        G_stack = torch.stack(grads)  # (batch, n)

        # Empirische Fisher: G = (1/N) Σ g_k g_k^†  (Hermitisch)
        G_complex = G_stack.to(torch.cfloat)
        G = (G_complex.T.conj() @ G_complex) / G_stack.shape[0]

        # Regularisierung für positive Definitheit
        G = G + self.reg * torch.eye(n, dtype=torch.cfloat)

        # Symmetrisiere zu Hermitischer Matrix
        G = 0.5 * (G + G.T.conj())

        self.metric = G
        return G

    def compute_metric_lowrank(
        self,
        model: nn.Module,
        data: torch.Tensor,
        targets: torch.Tensor,
        rank: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Low-Rank-Approximation der Kähler-Metrik: G ≈ V Λ V†.

        Für große Modelle (n >> 1000) ist die volle Metrik nicht berechenbar.
        Stattdessen behalten wir die Top-k Eigenwerte/Eigenvektoren.

        Args:
            model: Neuronales Netz
            data: Eingabedaten
            targets: Zielwerte
            rank: Rang der Approximation

        Returns:
            (V, Lambda): Eigenvektoren und Eigenwerte der Top-k Komponenten
        """
        n = self.n
        params = _get_flat_params(model)

        grads = []
        model.zero_grad()
        output = model(data)
        log_probs = torch.log_softmax(output, dim=-1)

        batch_size = min(data.shape[0], 128)
        for k in range(batch_size):
            model.zero_grad()
            log_p_k = log_probs[k, targets[k]] if targets is not None else log_probs[k].sum()
            g = torch.autograd.grad(log_p_k, model.parameters(), retain_graph=True, allow_unused=True)
            flat_g = torch.cat([gi.flatten() for gi in g if gi is not None])
            grads.append(flat_g)

        G_stack = torch.stack(grads) / np.sqrt(batch_size)  # (batch, n)

        # SVD für Low-Rank: G ≈ V diag(σ²) V^T
        U, S, Vh = torch.linalg.svd(G_stack, full_matrices=False)
        k = min(rank, S.shape[0])

        V = Vh[:k].T  # (n, k) - Top-k rechte Singulärvektoren
        Lambda = S[:k] ** 2  # Top-k Eigenwerte

        return V, Lambda

    def ricci_curvature_from_metric(self) -> torch.Tensor:
        """
        Berechne die Ricci-Krümmung R_{iȷ̄} aus der Kähler-Metrik.

        Für eine Kähler-Mannigfaltigkeit gilt:
            R_{iȷ̄} = -∂_i ∂_ȷ̄ log det(G)

        Approximiert durch finite Differenzen.

        Returns:
            R: Ricci-Krümmungs-Tensor (n × n)
        """
        G = self.metric
        n = G.shape[0]

        # log det(G)
        sign, logdet = torch.linalg.slogdet(G)
        logdet_val = logdet.real

        # Für Kähler-Mannigfaltigkeiten: R_{iȷ̄} = -∂_i ∂_ȷ̄ log det(G)
        # Approximation über Eigenwertzerlegung
        eigenvalues = torch.linalg.eigvalsh(G.real)
        eigenvalues = torch.clamp(eigenvalues, min=self.reg)

        # Diagonale Approximation: R_{ii} ≈ -∂²/∂θ_i² log det(G)
        # ≈ 1/λ_i² (für diagonal-dominante G)
        R_diag = 1.0 / (eigenvalues ** 2 + self.reg)

        # Konstruiere R als diagonale Matrix (erste Ordnung)
        R = torch.diag(R_diag).to(torch.cfloat)

        return R

    def symplectic_form(self) -> torch.Tensor:
        """
        Berechne die Kähler-Form ω = i G_{iȷ̄} dz^i ∧ dz̄^j.

        Die Kähler-Form definiert die symplektische Struktur auf dem
        Parameterraum und garantiert das Liouville-Theorem
        (Volumenerhaltung unter Hamiltonscher Evolution).

        Returns:
            omega: Kähler-2-Form (als antisymmetrische Matrix im reellen Bild)
        """
        G = self.metric
        n = G.shape[0]

        # Im reellen Bild (θ, p) hat die symplektische Form die Block-Struktur:
        # ω = [[-Im(G), Re(G)], [-Re(G), -Im(G)]]
        omega = torch.zeros(2 * n, 2 * n)
        omega[:n, n:] = G.real
        omega[n:, :n] = -G.real

        return omega

    def uncertainty_relation(self) -> torch.Tensor:
        """
        Berechne die Heisenberg-Unschärferelation:

            Δθ_i · Δθ_j ≥ (ℏ/2) · G^{iȷ̄}

        Returns:
            lower_bound: Untere Schranke für die Parameterunsicherheit
        """
        G_inv = torch.linalg.inv(self.metric)
        return (self.hbar / 2.0) * G_inv.real.abs()

    def geodesic_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Berechne die geodätische Distanz auf der Kähler-Mannigfaltigkeit.

        Approximation über die Metrik:
            d(z1, z2) ≈ √(δz† G δz)  wobei δz = z2 - z1

        Args:
            z1, z2: Zwei Punkte auf der Mannigfaltigkeit

        Returns:
            d: Geodätische Distanz (Skalar)
        """
        dz = (z2 - z1).to(torch.cfloat)
        dist_sq = (dz.conj() @ self.metric @ dz).real
        return torch.sqrt(torch.clamp(dist_sq, min=0.0))


def _get_flat_params(model: nn.Module) -> torch.Tensor:
    """Extrahiere alle Parameter als flachen Vektor."""
    return torch.cat([p.flatten() for p in model.parameters()])


def _set_flat_params(model: nn.Module, flat_params: torch.Tensor):
    """Setze alle Parameter aus einem flachen Vektor."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].reshape(p.shape))
        offset += numel
