"""
Quantum-Lift: Abbildung klassischer Gewichte auf kohärente Zustände im Fock-Raum.

Die klassischen Gewichte θ ∈ ℝⁿ werden auf kohärente Zustände im
bosonischen Fock-Raum über der Fisher-Metrik abgebildet:

    |θ⟩ = exp(-||θ||²/2) Σ_n (θⁿ/√n!) |n⟩

Kompression wird als Squeezing im Phasenraum implementiert:

    S(r) = exp(r/2 (a² - a†²))

Ein gesqueezter Zustand hat reduzierte Unsicherheit in einer Quadratur
auf Kosten erhöhter Unsicherheit in der konjugierten -- die quantenmechanische
Formulierung des Bias-Varianz-Tradeoffs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple


class QuantumLift:
    """
    Quantum-Lift-Layer: Hebt klassische Parameter in den Fock-Raum.

    Implementiert:
    1. Kohärente Zustände: |α⟩ = exp(-|α|²/2) Σ αⁿ/√n! |n⟩
    2. Squeeze-Operatoren: S(r) für Kompression
    3. Displacement-Operatoren: D(α) für Translation im Phasenraum
    4. Wigner-Funktion: Phasenraum-Darstellung der Quantenzustände
    """

    def __init__(
        self,
        n_params: int,
        fock_dim: int = 16,
        hbar: float = 0.1,
    ):
        """
        Args:
            n_params: Anzahl der klassischen Parameter
            fock_dim: Dimension des abgeschnittenen Fock-Raums
            hbar: Plancksches Wirkungsquantum (Unsicherheitsparameter)
        """
        self.n = n_params
        self.d = fock_dim
        self.hbar = hbar

        # Erzeugungs- und Vernichtungsoperatoren
        self.a = self._annihilation_operator()
        self.a_dag = self.a.T.conj()

        # Quadratur-Operatoren: q = √(ℏ/2)(a + a†), p = i√(ℏ/2)(a† - a)
        sqrt_hbar_2 = np.sqrt(hbar / 2)
        self.q_op = sqrt_hbar_2 * (self.a + self.a_dag)
        self.p_op = 1j * sqrt_hbar_2 * (self.a_dag - self.a)

    def _annihilation_operator(self) -> torch.Tensor:
        """Konstruiere den Vernichtungsoperator a im Fock-Raum."""
        d = self.d
        a = torch.zeros(d, d, dtype=torch.cfloat)
        for n in range(1, d):
            a[n - 1, n] = np.sqrt(n)
        return a

    def coherent_state(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Erzeuge kohärenten Zustand |α⟩ für einen einzelnen Parameter.

        |α⟩ = exp(-|α|²/2) Σ_{n=0}^{d-1} αⁿ/√n! |n⟩

        Args:
            alpha: Komplexe Amplitude (Skalar oder Vektor)

        Returns:
            state: Kohärenter Zustand im Fock-Raum (d,)
        """
        d = self.d
        alpha = alpha.to(torch.cfloat) if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.cfloat)

        # Fock-Koeffizienten
        state = torch.zeros(d, dtype=torch.cfloat)
        coeff = torch.tensor(1.0, dtype=torch.cfloat)  # α⁰/√0! = 1

        for n in range(d):
            state[n] = coeff
            if n < d - 1:
                coeff = coeff * alpha / np.sqrt(n + 1)

        # Normierung: exp(-|α|²/2)
        norm_factor = torch.exp(-0.5 * alpha.abs() ** 2)
        state = norm_factor * state

        return state

    def lift_parameters(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Hebe alle Parameter in den Fock-Raum.

        Jeder Parameter θ_i wird zu einem kohärenten Zustand |θ_i⟩.
        Der Gesamtzustand ist das Tensorprodukt: |Ψ⟩ = ⊗_i |θ_i⟩.

        In der Praxis speichern wir die Fock-Koeffizienten als Matrix (n × d).

        Args:
            theta: Parametervektor (n,)

        Returns:
            fock_states: Matrix der Fock-Koeffizienten (n × d)
        """
        n = min(theta.numel(), self.n)
        fock_states = torch.zeros(n, self.d, dtype=torch.cfloat)

        for i in range(n):
            fock_states[i] = self.coherent_state(theta[i])

        return fock_states

    def squeeze_operator(self, r: float) -> torch.Tensor:
        """
        Konstruiere den Squeeze-Operator S(r) im Fock-Raum.

        S(r) = exp(r/2 (a² - a†²))

        Ein gesqueezter Zustand hat:
        - Δq = √(ℏ/2) · e^{-r}  (reduzierte Unsicherheit in q)
        - Δp = √(ℏ/2) · e^{+r}  (erhöhte Unsicherheit in p)

        Dies ist die quantenmechanische Formulierung des Bias-Varianz-Tradeoffs:
        Mehr "Squeezen" = weniger Bias, mehr Varianz (oder umgekehrt).

        Args:
            r: Squeeze-Parameter (r > 0: squeeze q, r < 0: squeeze p)

        Returns:
            S: Squeeze-Operator (d × d Matrix)
        """
        d = self.d
        a = self.a
        a_dag = self.a_dag

        # Generator: G = (r/2)(a² - a†²)
        a_sq = a @ a
        a_dag_sq = a_dag @ a_dag
        G = (r / 2.0) * (a_sq - a_dag_sq)

        # Matrix-Exponential: S = exp(G)
        S = torch.matrix_exp(G)

        return S

    def displacement_operator(self, alpha: complex) -> torch.Tensor:
        """
        Konstruiere den Displacement-Operator D(α).

        D(α) = exp(α·a† - α*·a)

        Verschiebt den Zustand im Phasenraum um α.

        Args:
            alpha: Komplexe Verschiebung

        Returns:
            D: Displacement-Operator (d × d Matrix)
        """
        alpha_t = torch.tensor(alpha, dtype=torch.cfloat)
        G = alpha_t * self.a_dag - alpha_t.conj() * self.a
        D = torch.matrix_exp(G)
        return D

    def compress_via_squeezing(
        self,
        theta: torch.Tensor,
        squeeze_param: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Komprimiere Parameter durch Squeezing im Phasenraum.

        1. Hebe Parameter in den Fock-Raum
        2. Wende Squeeze-Operator an
        3. Projiziere zurück auf den niedrigsten Fock-Sektor
        4. Extrahiere komprimierte Parameter

        Args:
            theta: Originalparameter (n,)
            squeeze_param: Stärke des Squeezings (r > 0)

        Returns:
            (theta_compressed, stats): Komprimierte Parameter und Statistiken
        """
        n = min(theta.numel(), self.n)

        # 1. Lift in Fock-Raum
        fock_states = self.lift_parameters(theta[:n])

        # 2. Squeeze-Operator
        S = self.squeeze_operator(squeeze_param)

        # 3. Wende Squeezing an
        squeezed = (S @ fock_states.T).T  # (n, d)

        # 4. Projiziere auf klassische Parameter (Erwartungswert von q)
        # ⟨q⟩ = ⟨ψ|q_op|ψ⟩ für jeden Parameter
        theta_compressed = torch.zeros(n)
        uncertainties = torch.zeros(n)

        for i in range(n):
            psi = squeezed[i]
            psi_norm = psi / psi.norm().clamp(min=1e-15)

            # ⟨q⟩ = ψ† q_op ψ
            q_exp = (psi_norm.conj() @ self.q_op @ psi_norm).real
            theta_compressed[i] = q_exp

            # Δq = √(⟨q²⟩ - ⟨q⟩²)
            q2_exp = (psi_norm.conj() @ self.q_op @ self.q_op @ psi_norm).real
            var_q = q2_exp - q_exp ** 2
            uncertainties[i] = torch.sqrt(var_q.clamp(min=0))

        # Statistiken
        bias = (theta[:n] - theta_compressed).norm().item()
        variance = uncertainties.mean().item()

        stats = {
            "squeeze_parameter": squeeze_param,
            "bias": bias,
            "mean_uncertainty": variance,
            "uncertainty_product": variance * np.exp(squeeze_param) * np.sqrt(self.hbar / 2),
            "heisenberg_bound": self.hbar / 2,
            "compression_quality": 1.0 - bias / (theta[:n].norm().item() + 1e-10),
        }

        return theta_compressed, stats

    def wigner_function(
        self,
        state: torch.Tensor,
        q_range: Tuple[float, float] = (-3, 3),
        p_range: Tuple[float, float] = (-3, 3),
        resolution: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Berechne die Wigner-Funktion W(q, p) eines Quantenzustands.

        Die Wigner-Funktion ist die Phasenraum-Quasi-Wahrscheinlichkeitsverteilung:
            W(q, p) = (1/πℏ) Σ_{m,n} ρ_{mn} W_{mn}(q, p)

        Negative Werte der Wigner-Funktion zeigen "Quantenness" an.

        Args:
            state: Fock-Zustandsvektor (d,)
            q_range: Bereich der q-Achse
            p_range: Bereich der p-Achse
            resolution: Gitterpunkte pro Achse

        Returns:
            (q_grid, p_grid, W): Phasenraum-Gitter und Wigner-Funktion
        """
        d = self.d
        q_vals = torch.linspace(q_range[0], q_range[1], resolution)
        p_vals = torch.linspace(p_range[0], p_range[1], resolution)
        q_grid, p_grid = torch.meshgrid(q_vals, p_vals, indexing='ij')

        # Dichtematrix ρ = |ψ⟩⟨ψ|
        psi = state.to(torch.cfloat)
        rho = psi.unsqueeze(1) @ psi.unsqueeze(0).conj()

        # Wigner-Funktion über Laguerre-Polynome
        W = torch.zeros(resolution, resolution)

        for i in range(resolution):
            for j in range(resolution):
                q, p = q_vals[i].item(), p_vals[j].item()
                alpha = (q + 1j * p) / np.sqrt(2 * self.hbar)

                # Wigner via Displacement-Parity:
                # W(q,p) = (2/πℏ) Tr(D(α) Π D(-α) ρ)
                # Approximation über kohärente Zustandsüberlappung
                coh = self.coherent_state(torch.tensor(alpha))
                overlap = (coh.conj() @ rho @ coh).real
                W[i, j] = (2.0 / (np.pi * self.hbar)) * overlap

        return q_grid, p_grid, W

    def number_operator_expectation(self, state: torch.Tensor) -> float:
        """
        Erwartungswert des Besetzungszahl-Operators: ⟨n⟩ = ⟨a†a⟩.

        Gibt die mittlere "Energie" (Komplexität) des Zustands.

        Args:
            state: Fock-Zustandsvektor (d,)

        Returns:
            n_mean: Mittlere Besetzungszahl
        """
        psi = state.to(torch.cfloat)
        psi = psi / psi.norm().clamp(min=1e-15)
        n_op = self.a_dag @ self.a
        return (psi.conj() @ n_op @ psi).real.item()

    def thermal_state(self, n_mean: float) -> torch.Tensor:
        """
        Erzeuge einen thermischen Zustand mit mittlerer Besetzungszahl n_mean.

        ρ_thermal = Σ_n (n_mean^n / (1+n_mean)^{n+1}) |n⟩⟨n|

        Thermische Zustände repräsentieren maximale Unsicherheit (Maximum-Entropy)
        für gegebene mittlere Energie -- analog zu untrainierten Gewichten.

        Args:
            n_mean: Mittlere Besetzungszahl (Temperatur)

        Returns:
            rho: Thermische Dichtematrix (d × d)
        """
        d = self.d
        rho = torch.zeros(d, d, dtype=torch.cfloat)

        for n in range(d):
            p_n = (n_mean ** n) / ((1 + n_mean) ** (n + 1))
            rho[n, n] = p_n

        # Normierung
        rho = rho / torch.trace(rho).real.clamp(min=1e-15)
        return rho

    def fidelity(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """
        Berechne die Fidelity zwischen zwei Quantenzuständen.

        F = |⟨ψ₁|ψ₂⟩|²

        Args:
            state1, state2: Fock-Zustandsvektoren

        Returns:
            F: Fidelity (0 bis 1)
        """
        psi1 = state1.to(torch.cfloat) / state1.norm().clamp(min=1e-15)
        psi2 = state2.to(torch.cfloat) / state2.norm().clamp(min=1e-15)
        return (psi1.conj() @ psi2).abs().item() ** 2
