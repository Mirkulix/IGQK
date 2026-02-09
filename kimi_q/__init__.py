"""
KIMI-Q: Kähler Information Manifold Inverse-Quantum

Das erste holographische Neural-Collapse-System.

Kernkomponenten:
- KahlerManifold: Kähler-Fisher-Metrik auf komplexifiziertem Parameterraum
- HolographicRicciFlow: Quanten-Ricci-Fluss für geometrisches Pruning
- EntanglementPruner: Verschränkungs-basiertes Pruning mit Bell-Paar-Fusion
- HolographicEntropy: Ryu-Takayanagi-Entropiegrenzen
- DiracOperator: Spektrale Geometrie und topologischer Index
- QuantumLift: Kohärente Zustände im Fock-Raum
"""

__version__ = "0.1.0"

from kimi_q.core.kahler_manifold import KahlerManifold
from kimi_q.core.ricci_flow import HolographicRicciFlow
from kimi_q.core.entanglement_pruning import EntanglementPruner
from kimi_q.core.holographic import HolographicEntropy
from kimi_q.core.dirac_operator import DiracOperator
from kimi_q.core.quantum_lift import QuantumLift

__all__ = [
    "KahlerManifold",
    "HolographicRicciFlow",
    "EntanglementPruner",
    "HolographicEntropy",
    "DiracOperator",
    "QuantumLift",
]
