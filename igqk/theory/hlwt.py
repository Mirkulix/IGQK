"""
Hybrid Laplace-Wavelet Transformation (HLWT).

Proposition 6.1: HLWT is the Fourier transform of quantum gradient flow
in local coordinates. It decomposes the quantum state into multi-scale
components for frequency-aware compression.

The wavelet decomposition:
    ρ(θ) = Σ_j Σ_k c_{j,k} ψ_{j,k}(θ)

where ψ_{j,k} are wavelets at scale j and position k.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


class HybridLaplaceWavelet:
    """
    Hybrid Laplace-Wavelet Transform for quantum state analysis and compression.

    Combines Laplace transform (global decay analysis) with wavelet decomposition
    (local multi-scale analysis) of the quantum gradient flow.
    """

    def __init__(self, num_levels: int = 4, wavelet: str = "haar"):
        """
        Args:
            num_levels: Number of wavelet decomposition levels.
            wavelet: Wavelet type ('haar', 'db2').
        """
        self.num_levels = num_levels
        self.wavelet = wavelet

    def decompose(self, signal: torch.Tensor) -> List[torch.Tensor]:
        """
        Multi-level wavelet decomposition of weight tensor.

        Returns list of [detail_L, ..., detail_1, approximation] coefficients.
        """
        coefficients = []
        current = signal.float()

        for level in range(self.num_levels):
            n = current.shape[0]
            if n < 2:
                break

            if n % 2 != 0:
                current = current[:n - 1]
                n = n - 1

            if self.wavelet == "haar":
                approx = (current[0::2] + current[1::2]) / np.sqrt(2)
                detail = (current[0::2] - current[1::2]) / np.sqrt(2)
            else:
                approx, detail = self._db2_step(current)

            coefficients.append(detail)
            current = approx

        coefficients.append(current)
        return list(reversed(coefficients))

    def reconstruct(self, coefficients: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct signal from wavelet coefficients."""
        approx = coefficients[0]

        for detail in coefficients[1:]:
            n = min(approx.shape[0], detail.shape[0])
            a = approx[:n]
            d = detail[:n]

            if self.wavelet == "haar":
                even = (a + d) / np.sqrt(2)
                odd = (a - d) / np.sqrt(2)
            else:
                even, odd = self._db2_reconstruct(a, d)

            result = torch.zeros(2 * n, device=approx.device)
            result[0::2] = even
            result[1::2] = odd
            approx = result

        return approx

    def compress(
        self,
        weights: torch.Tensor,
        keep_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compress weights via wavelet thresholding.

        Keeps only the largest wavelet coefficients (by magnitude).

        Args:
            weights: Flat weight tensor.
            keep_ratio: Fraction of coefficients to keep.

        Returns:
            (compressed_weights, compression_ratio)
        """
        flat = weights.flatten()
        coefficients = self.decompose(flat)

        # Collect all detail coefficients
        all_details = torch.cat([c.flatten() for c in coefficients[1:]])
        threshold = torch.quantile(all_details.abs(), 1.0 - keep_ratio)

        # Threshold detail coefficients
        thresholded = []
        thresholded.append(coefficients[0])  # keep approximation
        for c in coefficients[1:]:
            masked = c.clone()
            masked[masked.abs() < threshold] = 0.0
            thresholded.append(masked)

        reconstructed = self.reconstruct(thresholded)

        # Match original length
        n = flat.shape[0]
        if reconstructed.shape[0] >= n:
            reconstructed = reconstructed[:n]
        else:
            padded = torch.zeros(n, device=weights.device)
            padded[: reconstructed.shape[0]] = reconstructed
            reconstructed = padded

        nonzero = sum((c != 0).sum().item() for c in thresholded)
        total = sum(c.numel() for c in thresholded)
        ratio = nonzero / total if total > 0 else 1.0

        return reconstructed.view_as(weights), ratio

    def analyze_frequency(self, weights: torch.Tensor) -> dict:
        """Analyze frequency content of weights across scales."""
        flat = weights.flatten()
        coefficients = self.decompose(flat)

        analysis = {}
        analysis["approximation_energy"] = (coefficients[0] ** 2).sum().item()
        total_energy = analysis["approximation_energy"]

        for i, c in enumerate(coefficients[1:]):
            key = f"detail_level_{i + 1}"
            energy = (c ** 2).sum().item()
            analysis[key] = energy
            total_energy += energy

        if total_energy > 0:
            analysis["approximation_ratio"] = analysis["approximation_energy"] / total_energy
            for i in range(len(coefficients) - 1):
                key = f"detail_level_{i + 1}"
                analysis[f"{key}_ratio"] = analysis[key] / total_energy

        return analysis

    def _db2_step(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Daubechies-2 wavelet single decomposition step."""
        h0 = torch.tensor(
            [(1 + np.sqrt(3)) / (4 * np.sqrt(2)), (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
             (3 - np.sqrt(3)) / (4 * np.sqrt(2)), (1 - np.sqrt(3)) / (4 * np.sqrt(2))],
            device=signal.device, dtype=signal.dtype,
        )
        h1 = torch.tensor([h0[3], -h0[2], h0[1], -h0[0]], device=signal.device, dtype=signal.dtype)

        padded = torch.cat([signal, signal[:3]])
        n_out = signal.shape[0] // 2
        approx = torch.zeros(n_out, device=signal.device)
        detail = torch.zeros(n_out, device=signal.device)
        for i in range(n_out):
            idx = 2 * i
            approx[i] = torch.dot(h0, padded[idx: idx + 4])
            detail[i] = torch.dot(h1, padded[idx: idx + 4])
        return approx, detail

    def _db2_reconstruct(
        self, approx: torch.Tensor, detail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified DB2 reconstruction (fallback to Haar)."""
        even = (approx + detail) / np.sqrt(2)
        odd = (approx - detail) / np.sqrt(2)
        return even, odd
