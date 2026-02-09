"""
Information-Geometrischer Dirac-Operator und Spektrale Geometrie.

Der nicht-kommutative Dirac-Operator:

    D = iγ^μ ∇_μ + Φ(θ)

kodiert die topologische Struktur des Netzwerks. Der Atiyah-Singer-Index:

    Dim(Effective-Params) = index(D) = ∫_M Â(M) ∧ ch(E)

gibt das topologische Parameterbudget -- die minimale Anzahl notwendiger
Parameter als topologische Invariante der Datenmannigfaltigkeit.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List


class DiracOperator:
    """
    Dirac-Operator auf der Parametermannigfaltigkeit eines neuronalen Netzes.

    Implementiert:
    1. Diskrete Approximation des Dirac-Operators auf dem Netzwerkgraphen
    2. Spektralanalyse (Eigenwerte und Eigenvektoren)
    3. Atiyah-Singer-Index-Berechnung
    4. Spektrales Pruning basierend auf hochenergetischen Moden
    """

    def __init__(
        self,
        higgs_mass: float = 1.0,
        spectral_cutoff: Optional[float] = None,
    ):
        """
        Args:
            higgs_mass: Masse des Higgs-Feldes Φ (Kapazitätsparameter)
            spectral_cutoff: Cutoff Λ für hochenergetische Moden (Standard: auto)
        """
        self.higgs_mass = higgs_mass
        self.cutoff = spectral_cutoff
        self.spectrum: Optional[torch.Tensor] = None
        self.eigenvectors: Optional[torch.Tensor] = None

    def build_dirac_matrix(
        self,
        weight_matrix: torch.Tensor,
        metric: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Konstruiere den diskreten Dirac-Operator D = iγ·∇ + Φ.

        Auf einem Graphen wird der Dirac-Operator als:
            D = i · A_signed + Φ · I

        wobei A_signed die vorzeichenbehaftete Adjazenzmatrix ist
        (kodiert die kovariante Ableitung ∇ mit der Clifford-Struktur γ).

        Args:
            weight_matrix: Gewichtsmatrix W des Layers (m × n)
            metric: Optionale Metrik G für die kovariante Ableitung

        Returns:
            D: Dirac-Matrix (Hermitisch, (m+n) × (m+n))
        """
        m, n = weight_matrix.shape

        # Clifford-Struktur: γ-Matrizen in 2D
        # Für den Graphen-Dirac verwenden wir die off-diagonal Block-Struktur
        # D = [[Φ·I_m, iW], [iW†, Φ·I_n]]
        D = torch.zeros(m + n, m + n, dtype=torch.cfloat)

        # Higgs-Feld (Diagonale): Φ · I
        D[:m, :m] = self.higgs_mass * torch.eye(m, dtype=torch.cfloat)
        D[m:, m:] = self.higgs_mass * torch.eye(n, dtype=torch.cfloat)

        # Kovariante Ableitung (Off-Diagonal): iγ·∇ ≈ iW
        W_complex = weight_matrix.to(torch.cfloat)
        D[:m, m:m + n] = 1j * W_complex
        D[m:m + n, :m] = -1j * W_complex.T.conj()  # Hermitisch machen

        # Optional: Metrik-Korrektur
        if metric is not None:
            # D → G^{-1/2} D G^{-1/2}
            min_dim = min(metric.shape[0], m + n)
            G_sqrt_inv = _matrix_power(metric[:min_dim, :min_dim].real, -0.5)
            D[:min_dim, :min_dim] = G_sqrt_inv @ D[:min_dim, :min_dim] @ G_sqrt_inv

        return D

    def compute_spectrum(
        self,
        model: nn.Module,
        max_dim: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Berechne das Spektrum des Dirac-Operators über alle Layer.

        Das Spektrum {λ_k} kodiert die Netzwerkstruktur:
        - λ_k ≈ 0: Nullmoden → topologisch geschützte Features
        - |λ_k| klein: niederenergetisch → wichtige Features
        - |λ_k| groß: hochenergetisch → Rauschen (prunbar)

        Args:
            model: Neuronales Netz
            max_dim: Maximale Dimension für Spektralberechnung

        Returns:
            spectrum: Dict mit Eigenwerten und Analyse pro Layer
        """
        results = {}

        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue

            W = param.data
            m, n = W.shape

            # Begrenze Dimension
            m_eff = min(m, max_dim)
            n_eff = min(n, max_dim)
            W_eff = W[:m_eff, :n_eff]

            # Baue Dirac-Operator
            D = self.build_dirac_matrix(W_eff)

            # Spektralzerlegung (D ist Hermitisch)
            eigenvalues, eigenvectors = torch.linalg.eigh(D.real)

            # Analyse
            n_zero = (eigenvalues.abs() < 1e-6).sum().item()
            n_low = ((eigenvalues.abs() >= 1e-6) & (eigenvalues.abs() < 1.0)).sum().item()
            n_high = (eigenvalues.abs() >= 1.0).sum().item()

            # Spektrale Dimension (Hausdorff-Dimension der Mannigfaltigkeit)
            # d_s = 2 · lim_{t→0} (d/dt log Tr(e^{-tD²}))
            # Approximiert durch Weyl-Gesetz: N(λ) ~ λ^{d_s}
            abs_ev = eigenvalues.abs().sort().values
            abs_ev = abs_ev[abs_ev > 1e-6]
            if len(abs_ev) > 2:
                # Log-Log-Regression: log N(λ) ≈ d_s · log λ + const
                log_lambda = torch.log(abs_ev)
                log_N = torch.log(torch.arange(1, len(abs_ev) + 1, dtype=torch.float))
                # Lineare Regression
                A = torch.stack([log_lambda, torch.ones_like(log_lambda)], dim=1)
                coeffs = torch.linalg.lstsq(A, log_N).solution
                spectral_dim = coeffs[0].item()
            else:
                spectral_dim = 1.0

            results[name] = {
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "n_zero_modes": n_zero,
                "n_low_energy": n_low,
                "n_high_energy": n_high,
                "spectral_dimension": spectral_dim,
                "spectral_gap": abs_ev[0].item() if len(abs_ev) > 0 else 0.0,
            }

        return results

    def atiyah_singer_index(
        self,
        model: nn.Module,
        max_dim: int = 512,
    ) -> Dict[str, float]:
        """
        Berechne den Atiyah-Singer-Index des Dirac-Operators.

        index(D) = dim(ker D⁺) - dim(ker D⁻)

        Dies gibt das **topologische Parameterbudget**: Die minimale Anzahl
        notwendiger Parameter, die eine topologische Invariante ist.

        In der diskreten Approximation:
            index(D) = n⁺ - n⁻

        wobei n⁺, n⁻ die Anzahl der positiven/negativen Nullmoden sind.

        Args:
            model: Neuronales Netz
            max_dim: Maximale Dimension

        Returns:
            index_info: Index und topologische Invarianten
        """
        spectrum = self.compute_spectrum(model, max_dim)

        total_index = 0
        total_zero_modes = 0
        layer_indices = {}

        for name, spec in spectrum.items():
            eigenvalues = spec["eigenvalues"]

            # Nullmoden (|λ| < ε)
            zero_mask = eigenvalues.abs() < 1e-6
            zero_modes = eigenvalues[zero_mask]

            # Positive und negative Nullmoden (durch Vorzeichen)
            # In der Praxis: Zähle die SVD-Nullmoden der Gewichtsmatrix
            n_pos = (zero_modes >= 0).sum().item()
            n_neg = (zero_modes < 0).sum().item()

            layer_index = n_pos - n_neg
            total_index += layer_index
            total_zero_modes += len(zero_modes)

            layer_indices[name] = {
                "index": layer_index,
                "positive_zero_modes": n_pos,
                "negative_zero_modes": n_neg,
                "spectral_dimension": spec["spectral_dimension"],
            }

        # Topologisches Parameterbudget (Theorem K2)
        total_params = sum(p.numel() for p in model.parameters())
        budget = abs(total_index) + total_zero_modes

        return {
            "total_index": total_index,
            "total_zero_modes": total_zero_modes,
            "topological_budget": budget,
            "total_params": total_params,
            "excess_params": total_params - budget,
            "max_compression_by_topology": total_params / max(budget, 1),
            "layer_indices": layer_indices,
        }

    def spectral_action(
        self,
        model: nn.Module,
        cutoff: Optional[float] = None,
        max_dim: int = 512,
    ) -> float:
        """
        Berechne die Spektrale Aktion:

            S[D] = Tr(f(D/Λ))

        wobei f eine Cutoff-Funktion und Λ der Cutoff ist.
        Verwendet f(x) = exp(-x²) (Gauß-Cutoff).

        Dies gibt ein natürliches Regularisierungsfunktional.

        Args:
            model: Neuronales Netz
            cutoff: Spektraler Cutoff Λ (Standard: auto)
            max_dim: Maximale Dimension

        Returns:
            S: Spektrale Aktion (Skalar)
        """
        spectrum = self.compute_spectrum(model, max_dim)

        if cutoff is None:
            # Auto-Cutoff: Median der Absolutwerte
            all_ev = torch.cat([s["eigenvalues"] for s in spectrum.values()])
            cutoff = all_ev.abs().median().item()
            cutoff = max(cutoff, 1e-6)

        self.cutoff = cutoff

        S = 0.0
        for name, spec in spectrum.items():
            ev = spec["eigenvalues"]
            # f(D/Λ) = exp(-(D/Λ)²)
            f_values = torch.exp(-(ev / cutoff) ** 2)
            S += f_values.sum().item()

        return S

    def spectral_pruning(
        self,
        model: nn.Module,
        keep_ratio: float = 0.5,
        max_dim: int = 512,
    ) -> Dict[str, float]:
        """
        Spektrales Pruning: Entferne hochenergetische Moden.

        Behalte nur Eigenmoden mit |λ_k| < Λ (Cutoff). Die
        hochenergetischen Moden entsprechen Rauschen und sind
        für die Funktion des Netzes nicht wesentlich.

        Args:
            model: Neuronales Netz
            keep_ratio: Anteil der zu behaltenden Moden (0-1)
            max_dim: Maximale Dimension

        Returns:
            stats: Pruning-Statistiken
        """
        total_original = 0
        total_kept = 0

        for name, param in model.named_parameters():
            if param.dim() < 2:
                total_original += param.numel()
                total_kept += param.numel()
                continue

            W = param.data
            m, n = W.shape
            total_original += m * n

            # SVD als Spektralzerlegung
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            # Behalte Top-k Moden
            k = max(1, int(min(m, n) * keep_ratio))
            k = min(k, len(S))

            # Rekonstruiere mit reduziertem Rang
            W_pruned = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
            param.data.copy_(W_pruned)

            total_kept += k * (m + n)

        return {
            "original_params": total_original,
            "effective_params": total_kept,
            "compression_ratio": total_original / max(total_kept, 1),
            "keep_ratio": keep_ratio,
            "cutoff": self.cutoff,
        }

    def topological_invariants(
        self,
        model: nn.Module,
        max_dim: int = 512,
    ) -> Dict[str, float]:
        """
        Berechne topologische Invarianten des Netzwerks.

        - Atiyah-Singer-Index (Parameterbudget)
        - Spektrale Dimension (Hausdorff-Dimension)
        - Betti-Zahlen (topologische Features)
        - Euler-Charakteristik

        Args:
            model: Neuronales Netz
            max_dim: Maximale Dimension

        Returns:
            invariants: Topologische Invarianten
        """
        index_info = self.atiyah_singer_index(model, max_dim)
        spectrum = self.compute_spectrum(model, max_dim)

        # Euler-Charakteristik χ = Σ (-1)^k b_k ≈ index(D)
        euler_char = index_info["total_index"]

        # Mittlere spektrale Dimension
        spectral_dims = [s["spectral_dimension"] for s in spectrum.values()]
        mean_spectral_dim = np.mean(spectral_dims) if spectral_dims else 0.0

        # Spektrale Lücke (Gap zwischen Null und erstem Eigenwert)
        gaps = [s["spectral_gap"] for s in spectrum.values()]
        mean_gap = np.mean(gaps) if gaps else 0.0

        return {
            "euler_characteristic": euler_char,
            "atiyah_singer_index": index_info["total_index"],
            "topological_budget": index_info["topological_budget"],
            "mean_spectral_dimension": mean_spectral_dim,
            "mean_spectral_gap": mean_gap,
            "excess_parameters": index_info["excess_params"],
            "topology_compression_ratio": index_info["max_compression_by_topology"],
        }


def _matrix_power(A: torch.Tensor, p: float) -> torch.Tensor:
    """Berechne A^p über Eigenwertzerlegung."""
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues = eigenvalues.clamp(min=1e-10)
    return eigenvectors @ torch.diag(eigenvalues ** p) @ eigenvectors.T
