"""
IGQK Plugin System - Extend IGQK with custom compression methods and hardware profiles.

Register custom:
- Compression methods
- Hardware profiles
- Measurement operators
- Annealing schedules
- Metrics

Usage:
    from igqk.plugins import PluginRegistry

    @PluginRegistry.register("compression", "my_method")
    def my_compression(weights, **kwargs):
        return my_quantize(weights)

    # Use registered plugin
    method = PluginRegistry.get("compression", "my_method")
    compressed = method(weights)

    # List all plugins
    PluginRegistry.list_plugins()
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PluginInfo:
    """Metadata about a registered plugin."""
    category: str
    name: str
    func: Callable
    description: str
    author: str
    version: str


class _PluginRegistry:
    """
    Central registry for IGQK plugins.

    Supports categories:
    - "compression": Custom compression methods
    - "hardware": Custom hardware profiles
    - "measurement": Custom measurement operators
    - "schedule": Custom annealing schedules
    - "metric": Custom evaluation metrics
    - "transform": Custom weight transforms
    """

    CATEGORIES = [
        "compression", "hardware", "measurement",
        "schedule", "metric", "transform",
    ]

    def __init__(self):
        self._plugins: Dict[str, Dict[str, PluginInfo]] = {
            cat: {} for cat in self.CATEGORIES
        }

    def register(
        self,
        category: str,
        name: str,
        description: str = "",
        author: str = "community",
        version: str = "1.0.0",
    ) -> Callable:
        """
        Register a plugin via decorator or direct call.

        Usage as decorator:
            @PluginRegistry.register("compression", "my_method")
            def my_method(weights, **kwargs):
                return quantized_weights

        Usage as direct call:
            PluginRegistry.register("compression", "my_method")(my_function)
        """
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Valid categories: {self.CATEGORIES}"
            )

        def decorator(func: Callable) -> Callable:
            self._plugins[category][name] = PluginInfo(
                category=category,
                name=name,
                func=func,
                description=description or func.__doc__ or "",
                author=author,
                version=version,
            )
            return func

        return decorator

    def get(self, category: str, name: str) -> Callable:
        """Get a registered plugin function."""
        if category not in self._plugins:
            raise ValueError(f"Unknown category: {category}")
        if name not in self._plugins[category]:
            available = list(self._plugins[category].keys())
            raise KeyError(
                f"Plugin '{name}' not found in '{category}'. "
                f"Available: {available}"
            )
        return self._plugins[category][name].func

    def get_info(self, category: str, name: str) -> PluginInfo:
        """Get plugin metadata."""
        if category not in self._plugins:
            raise ValueError(f"Unknown category: {category}")
        if name not in self._plugins[category]:
            raise KeyError(f"Plugin '{name}' not found in '{category}'")
        return self._plugins[category][name]

    def list_plugins(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """List all registered plugins, optionally filtered by category."""
        if category:
            if category not in self._plugins:
                raise ValueError(f"Unknown category: {category}")
            return {category: list(self._plugins[category].keys())}
        return {cat: list(plugins.keys()) for cat, plugins in self._plugins.items()}

    def has(self, category: str, name: str) -> bool:
        """Check if a plugin exists."""
        return (
            category in self._plugins
            and name in self._plugins[category]
        )

    def unregister(self, category: str, name: str):
        """Remove a registered plugin."""
        if category in self._plugins and name in self._plugins[category]:
            del self._plugins[category][name]

    def summary(self) -> str:
        """Get human-readable summary of all plugins."""
        lines = ["IGQK Plugin Registry", "=" * 50]
        total = 0
        for cat, plugins in self._plugins.items():
            if plugins:
                lines.append(f"\n  [{cat}] ({len(plugins)} plugins)")
                for name, info in plugins.items():
                    desc = info.description[:50] if info.description else "No description"
                    lines.append(f"    - {name} v{info.version} by {info.author}: {desc}")
                    total += 1
        lines.append(f"\nTotal: {total} plugins registered")
        return "\n".join(lines)


# Singleton instance
PluginRegistry = _PluginRegistry()


# --- Built-in plugins ---

@PluginRegistry.register(
    "compression", "ternary_basic",
    description="Basic ternary quantization to {-1, 0, +1}",
    author="IGQK", version="3.0.0",
)
def ternary_basic(weights, **kwargs):
    """Basic ternary quantization."""
    import torch
    std = weights.std()
    threshold = 0.7 * std
    result = torch.zeros_like(weights)
    result[weights > threshold] = std
    result[weights < -threshold] = -std
    return result


@PluginRegistry.register(
    "compression", "top_k_sparse",
    description="Keep top-k% of weights by magnitude",
    author="IGQK", version="3.0.0",
)
def top_k_sparse(weights, k=0.3, **kwargs):
    """Keep top k fraction of weights."""
    import torch
    flat = weights.flatten()
    num_keep = max(1, int(k * flat.numel()))
    _, indices = torch.topk(flat.abs(), num_keep)
    mask = torch.zeros_like(flat)
    mask[indices] = 1.0
    return (flat * mask).reshape(weights.shape)


@PluginRegistry.register(
    "compression", "uniform_quantize",
    description="Uniform quantization to n-bit",
    author="IGQK", version="3.0.0",
)
def uniform_quantize(weights, bits=4, **kwargs):
    """Quantize to uniform n-bit grid."""
    import torch
    n_levels = 2 ** bits
    w_min = weights.min()
    w_max = weights.max()
    scale = (w_max - w_min) / (n_levels - 1) if w_max != w_min else 1.0
    quantized = torch.round((weights - w_min) / scale) * scale + w_min
    return quantized


@PluginRegistry.register(
    "metric", "model_size_bytes",
    description="Calculate model size in bytes",
    author="IGQK", version="3.0.0",
)
def model_size_bytes(model, **kwargs):
    """Calculate total model size in bytes."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


@PluginRegistry.register(
    "metric", "sparsity_ratio",
    description="Calculate fraction of zero weights",
    author="IGQK", version="3.0.0",
)
def sparsity_ratio(model, **kwargs):
    """Calculate sparsity ratio."""
    total = sum(p.numel() for p in model.parameters())
    zeros = sum((p == 0).sum().item() for p in model.parameters())
    return zeros / total if total > 0 else 0
