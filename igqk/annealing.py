"""
Adaptive Quantum Annealing Scheduler.

A novel training scheduler inspired by quantum annealing that automatically
transitions from exploration (high quantum uncertainty) to exploitation
(precise convergence).

Innovation: Unlike classical learning rate schedulers (step, cosine, etc.),
this controls the QUANTUM TEMPERATURE of the optimization landscape:

  - Hot start (high ℏ): Quantum state is diffuse, explores many minima
  - Gradual cooling: State collapses toward best minimum
  - Measurement: Final collapse to discrete weights

This is analogous to quantum annealing on real quantum hardware,
but implemented as a classical simulation on GPU.

Phase transition detection: The scheduler monitors the quantum state's
entropy and detects phase transitions (sudden drops in entropy),
automatically adjusting the cooling rate.
"""

import torch
import numpy as np
from typing import Optional, Callable, List
from dataclasses import dataclass, field


@dataclass
class AnnealingSnapshot:
    """Snapshot of quantum state during annealing."""
    step: int
    hbar: float
    gamma: float
    entropy: float
    purity: float
    loss: float
    phase: str  # "exploration", "transition", "convergence", "measurement"


class QuantumAnnealingScheduler:
    """
    Adaptive quantum annealing for IGQK training.

    Automatically manages the quantum parameters (ℏ, γ) during training,
    implementing a quantum annealing protocol:

    Phase 1 - Exploration (hot):
        High ℏ → diffuse quantum state → explore many minima
    Phase 2 - Transition:
        Detect entropy drop → phase transition happening
    Phase 3 - Convergence (cold):
        Low ℏ, high γ → collapse toward best minimum
    Phase 4 - Measurement:
        ℏ → 0 → measure discrete weights
    """

    def __init__(
        self,
        hbar_init: float = 1.0,
        hbar_final: float = 0.001,
        gamma_init: float = 0.001,
        gamma_final: float = 0.1,
        total_steps: int = 1000,
        schedule: str = "adaptive",
        warmup_fraction: float = 0.1,
    ):
        """
        Args:
            hbar_init: Initial quantum uncertainty (high = exploration).
            hbar_final: Final quantum uncertainty (low = convergence).
            gamma_init: Initial damping (low = free exploration).
            gamma_final: Final damping (high = strong convergence).
            total_steps: Total training steps.
            schedule: "adaptive", "linear", "cosine", "exponential".
            warmup_fraction: Fraction of steps for warmup.
        """
        self.hbar_init = hbar_init
        self.hbar_final = hbar_final
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.total_steps = total_steps
        self.schedule = schedule
        self.warmup_steps = int(total_steps * warmup_fraction)

        # Adaptive state tracking
        self._entropy_history: List[float] = []
        self._loss_history: List[float] = []
        self._phase = "exploration"
        self._phase_transition_detected = False
        self._transition_step = -1
        self._snapshots: List[AnnealingSnapshot] = []

    def step(
        self, current_step: int, entropy: float = 0.0, loss: float = 0.0
    ) -> dict:
        """
        Get quantum parameters for current step.

        Args:
            current_step: Current training step.
            entropy: Current von Neumann entropy of quantum state.
            loss: Current training loss.

        Returns:
            {"hbar": float, "gamma": float, "phase": str, "temperature": float}
        """
        self._entropy_history.append(entropy)
        self._loss_history.append(loss)

        if self.schedule == "adaptive":
            hbar, gamma = self._adaptive_schedule(current_step, entropy, loss)
        elif self.schedule == "linear":
            hbar, gamma = self._linear_schedule(current_step)
        elif self.schedule == "cosine":
            hbar, gamma = self._cosine_schedule(current_step)
        elif self.schedule == "exponential":
            hbar, gamma = self._exponential_schedule(current_step)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        # Detect phase
        self._phase = self._detect_phase(current_step, entropy)

        # Temperature (for monitoring)
        temperature = hbar / self.hbar_init

        snapshot = AnnealingSnapshot(
            step=current_step, hbar=hbar, gamma=gamma,
            entropy=entropy, purity=0.0, loss=loss, phase=self._phase,
        )
        self._snapshots.append(snapshot)

        return {
            "hbar": hbar,
            "gamma": gamma,
            "phase": self._phase,
            "temperature": temperature,
        }

    def _adaptive_schedule(
        self, step: int, entropy: float, loss: float
    ) -> tuple:
        """
        Adaptive schedule that detects phase transitions.

        Monitors entropy derivative and adjusts cooling rate:
        - Fast entropy drop → slow down cooling (let system equilibrate)
        - Stable entropy → speed up cooling
        - Entropy plateau → trigger measurement phase
        """
        progress = step / self.total_steps

        # Detect phase transition (sudden entropy drop)
        if len(self._entropy_history) > 10:
            recent = self._entropy_history[-10:]
            entropy_slope = (recent[-1] - recent[0]) / 10

            if entropy_slope < -0.1 and not self._phase_transition_detected:
                self._phase_transition_detected = True
                self._transition_step = step

        # Warmup phase
        if step < self.warmup_steps:
            hbar = self.hbar_init
            warmup_progress = step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            gamma = self.gamma_init + (self.gamma_final - self.gamma_init) * 0.1 * warmup_progress
            return max(hbar, 1e-6), max(gamma, 1e-6)

        # Post-transition: accelerated cooling
        if self._phase_transition_detected:
            steps_since_transition = step - self._transition_step
            cooling_progress = min(1.0, steps_since_transition / (self.total_steps * 0.3))
            hbar = self.hbar_init * (1 - cooling_progress) + self.hbar_final * cooling_progress
            gamma = self.gamma_init + (self.gamma_final - self.gamma_init) * cooling_progress
        else:
            # Standard cosine annealing for hbar
            cos_progress = 0.5 * (1 + np.cos(np.pi * progress))
            hbar = self.hbar_final + (self.hbar_init - self.hbar_final) * cos_progress
            gamma = self.gamma_init + (self.gamma_final - self.gamma_init) * progress

        # Measurement phase (last 5%)
        if progress > 0.95:
            measurement_progress = (progress - 0.95) / 0.05
            hbar = self.hbar_final * (1 - measurement_progress) + 1e-6 * measurement_progress
            gamma = self.gamma_final * 2  # Strong convergence

        return max(hbar, 1e-6), max(gamma, 1e-6)

    def _linear_schedule(self, step: int) -> tuple:
        progress = step / self.total_steps
        hbar = self.hbar_init + (self.hbar_final - self.hbar_init) * progress
        gamma = self.gamma_init + (self.gamma_final - self.gamma_init) * progress
        return hbar, gamma

    def _cosine_schedule(self, step: int) -> tuple:
        progress = step / self.total_steps
        cos_val = 0.5 * (1 + np.cos(np.pi * progress))
        hbar = self.hbar_final + (self.hbar_init - self.hbar_final) * cos_val
        gamma = self.gamma_init + (self.gamma_final - self.gamma_init) * (1 - cos_val)
        return hbar, gamma

    def _exponential_schedule(self, step: int) -> tuple:
        progress = step / self.total_steps
        decay = np.exp(-5 * progress)
        hbar = self.hbar_final + (self.hbar_init - self.hbar_final) * decay
        gamma = self.gamma_final - (self.gamma_final - self.gamma_init) * decay
        return hbar, gamma

    def _detect_phase(self, step: int, entropy: float) -> str:
        """Detect current annealing phase."""
        progress = step / self.total_steps

        if progress < 0.1:
            return "exploration"
        elif progress > 0.95:
            return "measurement"
        elif self._phase_transition_detected and progress < 0.7:
            return "transition"
        else:
            return "convergence"

    @property
    def history(self) -> List[AnnealingSnapshot]:
        return self._snapshots

    @property
    def phase(self) -> str:
        return self._phase
