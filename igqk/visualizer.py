"""
Real-Time Quantum State Visualizer.

Live visualization of the quantum compression process:
- Density matrix eigenvalue evolution
- Entropy landscape
- Compression trajectory on manifold
- Phase transition detection
- Entanglement map

This makes the invisible quantum process VISIBLE,
turning abstract math into intuitive understanding.
"""

import torch
import numpy as np
from typing import Optional, List, Dict


class QuantumVisualizer:
    """
    Real-time visualization of quantum compression dynamics.

    Tracks the quantum state evolution and generates plots showing:
    1. How the quantum state collapses from diffuse → focused
    2. When phase transitions occur
    3. How entanglement between layers evolves
    4. The compression trajectory on the loss landscape
    """

    def __init__(self):
        self._steps: List[int] = []
        self._entropies: List[float] = []
        self._purities: List[float] = []
        self._losses: List[float] = []
        self._hbars: List[float] = []
        self._gammas: List[float] = []
        self._eigenvalue_history: List[List[float]] = []
        self._phases: List[str] = []
        self._grad_norms: List[float] = []

    def record(
        self,
        step: int,
        entropy: float = 0.0,
        purity: float = 0.0,
        loss: float = 0.0,
        hbar: float = 0.0,
        gamma: float = 0.0,
        eigenvalues: Optional[torch.Tensor] = None,
        phase: str = "",
        grad_norm: float = 0.0,
    ):
        """Record a snapshot of the quantum state."""
        self._steps.append(step)
        self._entropies.append(entropy)
        self._purities.append(purity)
        self._losses.append(loss)
        self._hbars.append(hbar)
        self._gammas.append(gamma)
        self._phases.append(phase)
        self._grad_norms.append(grad_norm)

        if eigenvalues is not None:
            self._eigenvalue_history.append(eigenvalues.detach().cpu().tolist())

    def plot_quantum_dashboard(self, save_path: Optional[str] = None):
        """
        Generate comprehensive quantum state dashboard.

        4-panel visualization:
        - Top-left: Loss + Entropy over time
        - Top-right: Eigenvalue spectrum evolution
        - Bottom-left: Quantum parameters (ℏ, γ) with phase annotations
        - Bottom-right: Purity trajectory (pure→mixed→pure)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("IGQK Quantum State Evolution", fontsize=16, fontweight="bold")

        steps = np.array(self._steps)

        # --- Top-left: Loss + Entropy ---
        ax = axes[0, 0]
        if self._losses:
            ax.plot(steps, self._losses, "b-", linewidth=2, label="Loss")
        ax.set_ylabel("Loss", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax2 = ax.twinx()
        if self._entropies:
            ax2.plot(steps, self._entropies, "r-", linewidth=2, label="Entropy S(ρ)")
        ax2.set_ylabel("von Neumann Entropy", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax.set_title("Loss & Quantum Entropy")
        ax.set_xlabel("Step")

        # --- Top-right: Eigenvalue spectrum ---
        ax = axes[0, 1]
        if self._eigenvalue_history:
            history = np.array(self._eigenvalue_history)
            for i in range(min(history.shape[1], 5)):
                ax.plot(steps[:len(history)], history[:, i],
                        linewidth=2, label=f"λ_{i+1}")
            ax.legend(fontsize=8)
        ax.set_title("Eigenvalue Spectrum Evolution")
        ax.set_xlabel("Step")
        ax.set_ylabel("Eigenvalue λ_i")

        # --- Bottom-left: Quantum parameters + phases ---
        ax = axes[1, 0]
        if self._hbars:
            ax.plot(steps, self._hbars, "g-", linewidth=2, label="ℏ (uncertainty)")
        ax.set_ylabel("ℏ", color="green")
        ax.tick_params(axis="y", labelcolor="green")
        ax3 = ax.twinx()
        if self._gammas:
            ax3.plot(steps, self._gammas, "m-", linewidth=2, label="γ (damping)")
        ax3.set_ylabel("γ", color="purple")
        ax3.tick_params(axis="y", labelcolor="purple")

        # Phase annotations
        if self._phases:
            phase_colors = {
                "exploration": "#FFE0B2",
                "transition": "#FFCDD2",
                "convergence": "#C8E6C9",
                "measurement": "#BBDEFB",
            }
            prev_phase = self._phases[0]
            start_idx = 0
            for i, phase in enumerate(self._phases):
                if phase != prev_phase or i == len(self._phases) - 1:
                    color = phase_colors.get(prev_phase, "#EEEEEE")
                    ax.axvspan(steps[start_idx], steps[min(i, len(steps)-1)],
                              alpha=0.2, color=color)
                    start_idx = i
                    prev_phase = phase

            legend_elements = [
                Patch(facecolor=c, alpha=0.3, label=p)
                for p, c in phase_colors.items()
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

        ax.set_title("Quantum Parameters & Annealing Phases")
        ax.set_xlabel("Step")

        # --- Bottom-right: Purity ---
        ax = axes[1, 1]
        if self._purities:
            ax.fill_between(steps, self._purities, alpha=0.3, color="#2196F3")
            ax.plot(steps, self._purities, "b-", linewidth=2)
        ax.set_ylim(0, 1.05)
        ax.set_title("State Purity Tr(ρ²)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Purity")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Pure state")
        ax.legend()

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_compression_analysis(
        self,
        original_weights: Dict[str, torch.Tensor],
        compressed_weights: Dict[str, torch.Tensor],
        save_path: Optional[str] = None,
    ):
        """Generate compression analysis visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_layers = min(len(original_weights), 6)
        fig, axes = plt.subplots(2, num_layers, figsize=(4 * num_layers, 8))
        fig.suptitle("IGQK Compression Analysis", fontsize=14, fontweight="bold")

        for idx, name in enumerate(list(original_weights.keys())[:num_layers]):
            orig = original_weights[name].flatten().numpy()
            comp = compressed_weights[name].flatten().numpy()

            # Original distribution
            ax = axes[0, idx] if num_layers > 1 else axes[0]
            ax.hist(orig, bins=50, alpha=0.7, color="#2196F3", edgecolor="black")
            ax.set_title(f"{name[:15]}\n(original)", fontsize=9)
            ax.set_ylabel("Count")

            # Compressed distribution
            ax = axes[1, idx] if num_layers > 1 else axes[1]
            ax.hist(comp, bins=50, alpha=0.7, color="#4CAF50", edgecolor="black")
            ax.set_title(f"(compressed)", fontsize=9)

            # Stats
            distortion = np.sum((orig - comp) ** 2)
            sparsity = np.mean(comp == 0) * 100
            ax.set_xlabel(f"D={distortion:.2f}, S={sparsity:.0f}%", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_entanglement_map(
        self,
        entanglement_data: dict,
        save_path: Optional[str] = None,
    ):
        """Visualize quantum entanglement between layers."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        layers = entanglement_data.get("layers", [])
        pairs = entanglement_data.get("pairs", {})

        if not layers:
            return None

        n = len(layers)
        matrix = np.zeros((n, n))
        layer_idx = {name: i for i, name in enumerate(layers)}

        for (a, b), info in pairs.items():
            i, j = layer_idx.get(a, -1), layer_idx.get(b, -1)
            if i >= 0 and j >= 0:
                mi = info["mutual_information"]
                matrix[i, j] = mi
                matrix[j, i] = mi

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

        short_names = [name[:12] for name in layers]
        ax.set_xticks(range(n))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(short_names, fontsize=8)

        fig.colorbar(im, ax=ax, label="Quantum Mutual Information I(A:B)")
        ax.set_title("Layer Entanglement Map", fontsize=14)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def summary(self) -> dict:
        """Get summary statistics of the recorded evolution."""
        return {
            "total_steps": len(self._steps),
            "final_entropy": self._entropies[-1] if self._entropies else 0,
            "final_purity": self._purities[-1] if self._purities else 0,
            "final_loss": self._losses[-1] if self._losses else 0,
            "min_loss": min(self._losses) if self._losses else 0,
            "entropy_reduction": (
                (self._entropies[0] - self._entropies[-1]) / self._entropies[0]
                if self._entropies and self._entropies[0] > 0
                else 0
            ),
            "phases_detected": list(set(self._phases)),
        }
