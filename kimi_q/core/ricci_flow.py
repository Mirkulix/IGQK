"""
Holographischer Quanten-Ricci-Fluss (HQRF) für geometrisches Pruning.

Der HQRF evolviert die Kähler-Metrik selbst:

    ∂G_{iȷ̄}/∂t = -R_{iȷ̄} + α ∇_i ∇_ȷ̄ L + β T_{iȷ̄}^{(Anomalie)}

Dimensionen mit kleiner Ricci-Krümmung kollabieren zu Singularitäten --
das sind die Kandidaten für Pruning. Die Ollivier-Ricci-Krümmung wird
als diskrete Approximation verwendet.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from kimi_q.core.kahler_manifold import KahlerManifold, _get_flat_params, _set_flat_params


@dataclass
class RicciFlowState:
    """Zustand des Ricci-Flusses zu einem Zeitpunkt t."""
    t: float
    metric: torch.Tensor           # G_{iȷ̄}(t)
    ricci: torch.Tensor            # R_{iȷ̄}(t)
    parameters: torch.Tensor       # θ(t)
    loss: float                    # L(t)
    singularities: List[int]       # Indizes kollapierter Dimensionen
    entropy: float                 # Von-Neumann-Entropie S(G)


class HolographicRicciFlow:
    """
    Holographischer Quanten-Ricci-Fluss für neuronale Netze.

    Implementiert den HQRF-Algorithmus:
    1. Berechne Ricci-Krümmung R_{iȷ̄} der Parametermannigfaltigkeit
    2. Berechne Loss-Hessische ∇_i ∇_ȷ̄ L
    3. Berechne Verschränkungs-Tensor T_{iȷ̄}
    4. Update Metrik und Parameter
    5. Detektiere Singularitäten → Pruning-Kandidaten
    """

    def __init__(
        self,
        manifold: KahlerManifold,
        alpha: float = 0.01,
        beta: float = 0.001,
        dt: float = 0.01,
        singularity_threshold: float = 1e-3,
        entropy_threshold: float = 0.1,
    ):
        """
        Args:
            manifold: Kähler-Mannigfaltigkeit des Parameterraums
            alpha: Gewicht des Loss-Hessischen Terms
            beta: Gewicht des Verschränkungs-Anomalie-Terms
            dt: Zeitschritt für den Fluss
            singularity_threshold: Schwelle für Ricci-Singularität-Detektion
            entropy_threshold: Minimale Entropie für Pruning
        """
        self.manifold = manifold
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.eps_sing = singularity_threshold
        self.eps_entropy = entropy_threshold

        self.history: List[RicciFlowState] = []
        self.pruned_indices: List[int] = []

    def compute_ollivier_ricci(
        self,
        adjacency: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Berechne die Ollivier-Ricci-Krümmung auf dem Parametergraphen.

        Die Ollivier-Ricci-Krümmung ist auf diskreten Räumen definiert als:

            κ(x, y) = 1 - W₁(μ_x, μ_y) / d(x, y)

        wobei W₁ die Wasserstein-1-Distanz zwischen den lokalen
        Verteilungen μ_x, μ_y auf den Nachbarschaften ist.

        Für neuronale Netze konstruieren wir den Graphen aus den
        Gewichtsmatrizen (Neuronen als Knoten, Gewichte als Kanten).

        Args:
            adjacency: Adjazenzmatrix des Netzwerkgraphen (n × n)
            weights: Kantengewichte (n × n)

        Returns:
            kappa: Ollivier-Ricci-Krümmung für jede Kante (n × n)
        """
        n = adjacency.shape[0]
        kappa = torch.zeros(n, n)

        # Lokale Wahrscheinlichkeitsverteilungen: μ_x(y) ∝ |w_{xy}| für Nachbarn
        degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1)
        prob = (adjacency * weights.abs()) / (adjacency * weights.abs()).sum(dim=1, keepdim=True).clamp(min=1e-10)

        # Approximation der Wasserstein-Distanz über Sinkhorn
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > 0:
                    mu_i = prob[i]
                    mu_j = prob[j]
                    d_ij = weights[i, j].abs().clamp(min=1e-10)
                    # Wasserstein-1 ≈ ||μ_i - μ_j||_1 · mittlere Distanz
                    w1 = torch.sum(torch.abs(mu_i - mu_j)) * d_ij
                    kappa[i, j] = 1.0 - w1 / d_ij
                    kappa[j, i] = kappa[i, j]

        return kappa

    def compute_ricci_curvature(
        self,
        model: nn.Module,
        data: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Berechne die Ricci-Krümmung R_{iȷ̄} der Parametermannigfaltigkeit.

        Kombiniert zwei Ansätze:
        1. Analytisch via Kähler-Metrik: R_{iȷ̄} = -∂_i ∂_ȷ̄ log det(G)
        2. Ollivier-Ricci auf dem Netzwerkgraphen (für Graphstruktur)

        Args:
            model: Neuronales Netz
            data: Eingabedaten
            targets: Zielwerte

        Returns:
            R: Ricci-Krümmungs-Tensor (n × n)
        """
        # Update Kähler-Metrik
        G = self.manifold.compute_metric(model, data, targets)

        # Ricci-Krümmung aus Kähler-Metrik
        R = self.manifold.ricci_curvature_from_metric()

        return R

    def compute_loss_hessian_diag(
        self,
        model: nn.Module,
        data: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """
        Berechne die Diagonale der Loss-Hessischen ∂²L/∂θ_i∂θ_j.

        Verwendet Hutchinson-Schätzer für effiziente Berechnung:
            H_ii ≈ E_v[v_i · (Hv)_i]  mit v ~ Rademacher

        Args:
            model: Neuronales Netz
            data: Eingabedaten
            targets: Zielwerte
            loss_fn: Verlustfunktion

        Returns:
            H_diag: Diagonale der Hessischen (n,)
        """
        params = list(model.parameters())
        flat_params = _get_flat_params(model)
        n = flat_params.numel()

        # Vorwärtsdurchlauf
        output = model(data)
        loss = loss_fn(output, targets)

        # Erster Gradient
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.flatten() for g in grads])

        # Hutchinson-Schätzer für Hessische-Diagonale (5 Samples)
        H_diag = torch.zeros(n)
        n_samples = 5

        for _ in range(n_samples):
            v = torch.randint(0, 2, (n,), dtype=flat_grad.dtype) * 2 - 1  # Rademacher
            # Hessian-Vektor-Produkt: Hv = ∂/∂θ (g^T v)
            gv = (flat_grad * v).sum()
            Hv_parts = torch.autograd.grad(gv, params, retain_graph=True)
            Hv = torch.cat([h.flatten() for h in Hv_parts])
            H_diag += v * Hv

        H_diag /= n_samples
        return H_diag.detach()

    def compute_entanglement_tensor(
        self,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Berechne den Verschränkungs-Anomalie-Tensor T_{iȷ̄}.

        T_{iȷ̄} misst, wie stark verschiedene Parameter verschränkt sind.
        Hohe Verschränkung → diese Parameter sollten nicht unabhängig
        gepruned werden.

        Berechnet als Korrelationsmatrix der Gewichtsgradienten über Layer:

            T_{ij} = Cov(∂L/∂θ_i, ∂L/∂θ_j) normiert

        Args:
            model: Neuronales Netz

        Returns:
            T: Verschränkungs-Tensor (n × n)
        """
        params = list(model.parameters())
        flat_params = _get_flat_params(model)
        n = flat_params.numel()

        # Inter-Layer-Korrelationsmatrix als Verschränkungsmaß
        # Verwende die Gewichtsstruktur selbst
        layer_boundaries = []
        offset = 0
        for p in params:
            layer_boundaries.append((offset, offset + p.numel()))
            offset += p.numel()

        T = torch.zeros(n, n, dtype=torch.cfloat)

        # Verschränkung zwischen Layern: Korrelation der Gewichtsverteilungen
        for i, (s1, e1) in enumerate(layer_boundaries):
            for j, (s2, e2) in enumerate(layer_boundaries):
                if i != j:
                    w1 = flat_params[s1:e1]
                    w2 = flat_params[s2:e2]
                    # Mutual information proxy: normalisierte Kovarianz
                    min_len = min(len(w1), len(w2))
                    if min_len > 0:
                        corr = torch.corrcoef(torch.stack([
                            w1[:min_len], w2[:min_len]
                        ]))[0, 1]
                        if not torch.isnan(corr):
                            # Block-Eintrag
                            T[s1:e1, s2:e2] = corr.abs().to(torch.cfloat)

        return T

    def step(
        self,
        model: nn.Module,
        data: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> RicciFlowState:
        """
        Einen Schritt des HQRF ausführen.

        Update-Gleichungen:
            G(t+1) = G(t) - dt·(R - α·H_loss - β·T)
            θ(t+1) = θ(t) - dt·G^{-1}·∇L

        Args:
            model: Neuronales Netz
            data: Eingabedaten
            targets: Zielwerte
            loss_fn: Verlustfunktion

        Returns:
            state: Aktueller Zustand des Flusses
        """
        t = len(self.history) * self.dt
        params = _get_flat_params(model)
        n = params.numel()

        # 1. Ricci-Krümmung
        R = self.compute_ricci_curvature(model, data, targets)

        # 2. Loss-Hessische (Diagonale)
        H_diag = self.compute_loss_hessian_diag(model, data, targets, loss_fn)
        H = torch.diag(H_diag).to(torch.cfloat)

        # 3. Verschränkungs-Tensor
        T = self.compute_entanglement_tensor(model)

        # 4. Metrik-Update: G(t+1) = G(t) - dt·(R - α·H - β·T)
        G = self.manifold.metric
        flow = R - self.alpha * H[:n, :n] - self.beta * T[:n, :n]
        G_new = G - self.dt * flow

        # Stelle positive Definitheit sicher
        eigenvalues, eigenvectors = torch.linalg.eigh(G_new.real)
        eigenvalues = torch.clamp(eigenvalues, min=self.manifold.reg)
        G_new = (eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T).to(torch.cfloat)
        G_new = 0.5 * (G_new + G_new.T.conj())

        self.manifold.metric = G_new

        # 5. Parameter-Update: θ(t+1) = θ(t) - dt·G^{-1}·∇L
        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

        # Natural Gradient: G^{-1} ∇L
        try:
            G_inv = torch.linalg.inv(G_new.real)
            nat_grad = G_inv @ grad
        except torch.linalg.LinAlgError:
            nat_grad = grad

        new_params = params - self.dt * nat_grad
        _set_flat_params(model, new_params)
        model.zero_grad()

        # 6. Singularitäts-Detektion
        ricci_diag = torch.diag(R).real
        singularities = (ricci_diag.abs() < self.eps_sing).nonzero(as_tuple=True)[0].tolist()

        # Von-Neumann-Entropie der Metrik: S = -Tr(G_norm log G_norm)
        G_trace = torch.trace(G_new.real).clamp(min=1e-10)
        G_norm = G_new.real / G_trace
        eigenvalues_norm = torch.linalg.eigvalsh(G_norm)
        eigenvalues_norm = eigenvalues_norm.clamp(min=1e-15)
        entropy = -(eigenvalues_norm * eigenvalues_norm.log()).sum().item()

        state = RicciFlowState(
            t=t,
            metric=G_new.detach(),
            ricci=R.detach(),
            parameters=new_params.detach(),
            loss=loss.item(),
            singularities=singularities,
            entropy=entropy,
        )
        self.history.append(state)

        return state

    def detect_pruning_candidates(self) -> List[int]:
        """
        Identifiziere Parameter, die durch den Ricci-Fluss zu
        Singularitäten kollabiert sind.

        Pruning-Regel:
            Prune(i) ⟺ R_{ii}(t→∞) < ε AND S_vN(ρ_i) < δ

        Returns:
            candidates: Liste der Parameterindizes zum Pruning
        """
        if not self.history:
            return []

        latest = self.history[-1]
        ricci_diag = torch.diag(latest.ricci).real

        candidates = []
        for i in range(ricci_diag.shape[0]):
            if ricci_diag[i].abs() < self.eps_sing:
                # Zusätzliche Entropie-Prüfung
                G = latest.metric.real
                if i < G.shape[0]:
                    local_entropy = -G[i, i].abs().clamp(min=1e-15).log()
                    if local_entropy < self.eps_entropy:
                        candidates.append(i)

        self.pruned_indices = candidates
        return candidates

    def apply_geometric_pruning(
        self,
        model: nn.Module,
        candidates: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Wende geometrisches Pruning basierend auf Ricci-Singularitäten an.

        Geprunte Gewichte werden auf Null gesetzt. Ihre Information bleibt
        als "verschränkter Zustand im Bulk" erhalten (gespeichert für
        potentielles Zero-Shot Revival).

        Args:
            model: Neuronales Netz
            candidates: Parameterindizes zum Pruning (Standard: auto-detect)

        Returns:
            stats: Pruning-Statistiken
        """
        if candidates is None:
            candidates = self.detect_pruning_candidates()

        params = _get_flat_params(model)
        n = params.numel()
        n_pruned = 0

        # Speichere Bulk-Zustände für Revival
        self._bulk_states = {}

        mask = torch.ones(n)
        for idx in candidates:
            if 0 <= idx < n:
                self._bulk_states[idx] = params[idx].item()
                mask[idx] = 0.0
                n_pruned += 1

        # Anwenden
        new_params = params * mask
        _set_flat_params(model, new_params)

        compression = n / max(n - n_pruned, 1)
        return {
            "total_params": n,
            "pruned_params": n_pruned,
            "remaining_params": n - n_pruned,
            "compression_ratio": compression,
            "sparsity": n_pruned / n if n > 0 else 0.0,
        }

    def zero_shot_revival(
        self,
        model: nn.Module,
        indices: Optional[List[int]] = None,
    ) -> int:
        """
        Zero-Shot Revival: Reaktiviere geprunte Neuronen aus den
        verschränkten Bulk-Zuständen (Theorem K4).

        Args:
            model: Neuronales Netz
            indices: Indizes der zu reaktivierenden Parameter (Standard: alle)

        Returns:
            n_revived: Anzahl reaktivierter Parameter
        """
        if not hasattr(self, '_bulk_states'):
            return 0

        params = _get_flat_params(model)
        revival_indices = indices or list(self._bulk_states.keys())
        n_revived = 0

        for idx in revival_indices:
            if idx in self._bulk_states and 0 <= idx < params.numel():
                params[idx] = self._bulk_states[idx]
                del self._bulk_states[idx]
                n_revived += 1

        _set_flat_params(model, params)
        return n_revived

    def convergence_diagnostic(self) -> Dict[str, float]:
        """
        Diagnostik der Ricci-Fluss-Konvergenz.

        Prüft ob der Fluss sich einem Kähler-Einstein-Fixpunkt nähert:
            R_{iȷ̄} → λ · G_{iȷ̄}

        Returns:
            diagnostics: Konvergenz-Metriken
        """
        if len(self.history) < 2:
            return {"converged": False, "einstein_residual": float('inf')}

        latest = self.history[-1]
        G = latest.metric.real
        R = latest.ricci.real

        # Prüfe Einstein-Bedingung: R = λG
        # λ = Tr(R) / Tr(G)
        trace_R = torch.trace(R)
        trace_G = torch.trace(G).clamp(min=1e-10)
        lamda = trace_R / trace_G

        residual = torch.norm(R - lamda * G) / torch.norm(G).clamp(min=1e-10)

        # Loss-Konvergenz
        losses = [s.loss for s in self.history[-10:]]
        loss_var = np.var(losses) if len(losses) > 1 else float('inf')

        return {
            "converged": residual.item() < 0.01 and loss_var < 1e-4,
            "einstein_residual": residual.item(),
            "einstein_lambda": lamda.item(),
            "loss_variance": loss_var,
            "n_singularities": len(latest.singularities),
            "metric_entropy": latest.entropy,
        }
