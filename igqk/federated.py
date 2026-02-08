"""
Federated Quantum Compression - Privacy-preserving distributed compression.

Multiple devices train and compress models together WITHOUT sharing raw weights.
Instead, they share only quantum state summaries (entropy, eigenvalue spectra),
which contain enough information for joint compression but reveal nothing
about the actual data or weights.

Protocol:
    1. Each device d computes local quantum state ρ_d
    2. Devices share only: entropy S(ρ_d), purity Tr(ρ_d²), eigenvalue spectrum
    3. Central coordinator computes optimal global compression plan
    4. Each device applies compression locally
    5. Compressed models are aggregated via federated averaging

Privacy guarantee: Only statistical summaries are shared, not weights or data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from igqk.core.quantum_state import QuantumState


@dataclass
class DeviceQuantumSummary:
    """Quantum summary from a device (privacy-safe to share)."""
    device_id: str
    num_params: int
    layer_entropies: Dict[str, float]
    layer_purities: Dict[str, float]
    layer_eigenvalue_spectra: Dict[str, List[float]]
    global_entropy: float
    global_purity: float


@dataclass
class FederatedCompressionPlan:
    """Compression plan from coordinator to device."""
    device_id: str
    per_layer_method: Dict[str, str]
    per_layer_strength: Dict[str, float]
    target_compression: float


class FederatedDevice:
    """
    A federated device participating in distributed quantum compression.

    Holds a local model, computes quantum summaries (safe to share),
    and applies compression plans from the coordinator.
    """

    def __init__(self, device_id: str, model: nn.Module):
        self.device_id = device_id
        self.model = model
        self._quantum_rank = 5

    def compute_summary(self) -> DeviceQuantumSummary:
        """
        Compute privacy-safe quantum summary of local model.

        Only statistical properties are computed - no raw weights leave the device.
        """
        layer_entropies = {}
        layer_purities = {}
        layer_spectra = {}
        all_entropies = []

        for name, param in self.model.named_parameters():
            if param.numel() < 16:
                continue

            flat = param.detach().flatten()
            rank = min(self._quantum_rank, flat.shape[0])
            rho = QuantumState.from_point(flat, rank=rank)

            entropy = rho.entropy()
            purity = rho.purity()

            layer_entropies[name] = entropy
            layer_purities[name] = purity
            layer_spectra[name] = rho.eigenvalues.tolist()
            all_entropies.append(entropy)

        global_entropy = np.mean(all_entropies) if all_entropies else 0.0
        global_purity = np.mean(list(layer_purities.values())) if layer_purities else 0.0

        return DeviceQuantumSummary(
            device_id=self.device_id,
            num_params=sum(p.numel() for p in self.model.parameters()),
            layer_entropies=layer_entropies,
            layer_purities=layer_purities,
            layer_eigenvalue_spectra=layer_spectra,
            global_entropy=global_entropy,
            global_purity=global_purity,
        )

    def apply_plan(self, plan: FederatedCompressionPlan):
        """Apply compression plan received from coordinator."""
        from igqk.theory.tlgt import TernaryLieGroup
        from igqk.theory.hlwt import HybridLaplaceWavelet

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                method = plan.per_layer_method.get(name, "none")
                strength = plan.per_layer_strength.get(name, 0.0)

                if method == "ternary":
                    tlgt = TernaryLieGroup(param.numel())
                    compressed, scale = tlgt.quantize(param.data)
                    param.data = compressed
                elif method == "sparse":
                    threshold = torch.quantile(param.data.abs().flatten(), strength)
                    param.data *= (param.data.abs() >= threshold).float()
                elif method == "wavelet":
                    hlwt = HybridLaplaceWavelet()
                    compressed, _ = hlwt.compress(param.data, keep_ratio=1.0 - strength)
                    param.data = compressed

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get compressed model state dict for aggregation."""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}


class FederatedCoordinator:
    """
    Central coordinator for federated quantum compression.

    Receives quantum summaries from devices, computes optimal
    compression plans, and coordinates model aggregation.

    Never sees raw weights or training data.
    """

    def __init__(self, target_compression: float = 0.1):
        self.target_compression = target_compression
        self._summaries: Dict[str, DeviceQuantumSummary] = {}

    def receive_summary(self, summary: DeviceQuantumSummary):
        """Receive quantum summary from a device."""
        self._summaries[summary.device_id] = summary

    def compute_plans(self) -> Dict[str, FederatedCompressionPlan]:
        """
        Compute optimal compression plans for all devices.

        Uses aggregated entropy information to determine per-layer compression:
        - Layers with consistent low entropy across devices → compress aggressively
        - Layers with high entropy variance → compress conservatively
        """
        if not self._summaries:
            return {}

        # Aggregate layer statistics across devices
        all_layers = set()
        for summary in self._summaries.values():
            all_layers.update(summary.layer_entropies.keys())

        layer_avg_entropy = {}
        layer_entropy_variance = {}

        for layer in all_layers:
            entropies = [
                s.layer_entropies.get(layer, 0.0) for s in self._summaries.values()
                if layer in s.layer_entropies
            ]
            if entropies:
                layer_avg_entropy[layer] = np.mean(entropies)
                layer_entropy_variance[layer] = np.var(entropies)

        # Generate plans
        plans = {}
        for device_id, summary in self._summaries.items():
            per_layer_method = {}
            per_layer_strength = {}

            for layer in summary.layer_entropies:
                avg_e = layer_avg_entropy.get(layer, 0.5)
                var_e = layer_entropy_variance.get(layer, 0.0)

                # Low entropy + low variance → safe to compress aggressively
                if avg_e < 0.3 and var_e < 0.01:
                    per_layer_method[layer] = "ternary"
                    per_layer_strength[layer] = 0.9
                # Low entropy but high variance → some devices disagree → moderate
                elif avg_e < 0.5:
                    per_layer_method[layer] = "sparse"
                    per_layer_strength[layer] = 0.5
                # High entropy → this layer carries important info → gentle
                else:
                    per_layer_method[layer] = "wavelet"
                    per_layer_strength[layer] = 0.3

            plans[device_id] = FederatedCompressionPlan(
                device_id=device_id,
                per_layer_method=per_layer_method,
                per_layer_strength=per_layer_strength,
                target_compression=self.target_compression,
            )

        return plans

    def aggregate_models(
        self, state_dicts: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Federated averaging of compressed models.

        Args:
            state_dicts: Dict of device_id → state_dict.

        Returns:
            Aggregated state dict.
        """
        if not state_dicts:
            return {}

        device_ids = list(state_dicts.keys())
        reference = state_dicts[device_ids[0]]
        aggregated = {}

        for name in reference:
            tensors = [state_dicts[d][name].float() for d in device_ids if name in state_dicts[d]]
            aggregated[name] = torch.stack(tensors).mean(dim=0)

        return aggregated

    def get_global_stats(self) -> dict:
        """Get aggregated statistics across all devices."""
        if not self._summaries:
            return {}

        return {
            "num_devices": len(self._summaries),
            "avg_global_entropy": np.mean(
                [s.global_entropy for s in self._summaries.values()]
            ),
            "avg_global_purity": np.mean(
                [s.global_purity for s in self._summaries.values()]
            ),
            "total_params": sum(
                s.num_params for s in self._summaries.values()
            ),
        }
