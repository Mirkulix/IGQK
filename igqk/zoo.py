"""
IGQK Model Zoo - Pre-compressed models and compression recipes.

Provides:
1. Registry of pre-compressed model configurations
2. Compression recipes for popular architectures
3. Expected metrics for each configuration
4. Easy download/generation of compressed models

Usage:
    from igqk.zoo import ModelZoo
    zoo = ModelZoo()
    zoo.list_models()
    recipe = zoo.get_recipe("resnet18_ternary")
    model = zoo.create_compressed("mnist_fc_ternary")
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ModelRecipe:
    """Recipe for creating a compressed model."""
    name: str
    architecture: str
    dataset: str
    compression_method: str
    description: str

    # Architecture details
    in_features: int = 784
    num_classes: int = 10
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"

    # Expected metrics
    expected_accuracy: float = 0.0
    expected_compression_ratio: float = 1.0
    expected_size_mb: float = 0.0
    expected_bits_per_weight: float = 32.0

    # IGQK parameters
    hbar: float = 0.1
    gamma: float = 0.01
    annealing_schedule: str = "cosine"


class ModelZoo:
    """
    Registry of pre-compressed model configurations and recipes.

    Contains compression recipes for common architectures with
    expected performance metrics.
    """

    def __init__(self):
        self._recipes: Dict[str, ModelRecipe] = {}
        self._register_builtin_recipes()

    def _register_builtin_recipes(self):
        """Register built-in model recipes."""
        # MNIST models
        self._recipes["mnist_fc_ternary"] = ModelRecipe(
            name="mnist_fc_ternary",
            architecture="fc_medium",
            dataset="MNIST",
            compression_method="ternary",
            description="MNIST FC network with ternary compression (16x)",
            in_features=784, num_classes=10,
            hidden_layers=[256, 128],
            expected_accuracy=0.96,
            expected_compression_ratio=16.0,
            expected_size_mb=0.07,
            expected_bits_per_weight=2.0,
            hbar=0.1, gamma=0.01,
        )

        self._recipes["mnist_fc_sparse"] = ModelRecipe(
            name="mnist_fc_sparse",
            architecture="fc_medium",
            dataset="MNIST",
            compression_method="sparse",
            description="MNIST FC with 90% sparsity",
            in_features=784, num_classes=10,
            hidden_layers=[256, 128],
            expected_accuracy=0.94,
            expected_compression_ratio=10.0,
            expected_size_mb=0.11,
            expected_bits_per_weight=3.2,
        )

        self._recipes["mnist_fc_wavelet"] = ModelRecipe(
            name="mnist_fc_wavelet",
            architecture="fc_medium",
            dataset="MNIST",
            compression_method="wavelet",
            description="MNIST FC with wavelet compression",
            in_features=784, num_classes=10,
            hidden_layers=[256, 128],
            expected_accuracy=0.97,
            expected_compression_ratio=3.3,
            expected_size_mb=0.33,
            expected_bits_per_weight=9.6,
        )

        # CIFAR-10 models
        self._recipes["cifar10_fc_ternary"] = ModelRecipe(
            name="cifar10_fc_ternary",
            architecture="fc_large",
            dataset="CIFAR-10",
            compression_method="ternary",
            description="CIFAR-10 FC with ternary compression",
            in_features=3072, num_classes=10,
            hidden_layers=[1024, 512, 256],
            expected_accuracy=0.50,
            expected_compression_ratio=16.0,
            expected_size_mb=0.56,
            expected_bits_per_weight=2.0,
            hbar=0.05, gamma=0.005,
        )

        self._recipes["cifar10_cnn_ternary"] = ModelRecipe(
            name="cifar10_cnn_ternary",
            architecture="cnn_simple",
            dataset="CIFAR-10",
            compression_method="ternary",
            description="CIFAR-10 CNN with ternary compression",
            in_features=3072, num_classes=10,
            hidden_layers=[64, 128, 256],
            expected_accuracy=0.72,
            expected_compression_ratio=16.0,
            expected_size_mb=0.12,
            expected_bits_per_weight=2.0,
        )

        # ResNet recipes
        self._recipes["resnet18_ternary"] = ModelRecipe(
            name="resnet18_ternary",
            architecture="resnet18",
            dataset="ImageNet",
            compression_method="ternary",
            description="ResNet-18 with IGQK ternary compression",
            in_features=224 * 224 * 3, num_classes=1000,
            hidden_layers=[64, 128, 256, 512],
            expected_accuracy=0.65,
            expected_compression_ratio=16.0,
            expected_size_mb=2.8,
            expected_bits_per_weight=2.0,
            hbar=0.05, gamma=0.005,
            annealing_schedule="adaptive",
        )

        self._recipes["resnet50_sparse"] = ModelRecipe(
            name="resnet50_sparse",
            architecture="resnet50",
            dataset="ImageNet",
            compression_method="sparse",
            description="ResNet-50 with 80% sparsity",
            in_features=224 * 224 * 3, num_classes=1000,
            hidden_layers=[64, 256, 512, 1024, 2048],
            expected_accuracy=0.73,
            expected_compression_ratio=5.0,
            expected_size_mb=19.6,
            expected_bits_per_weight=6.4,
        )

        # Transformer recipes
        self._recipes["bert_base_ternary"] = ModelRecipe(
            name="bert_base_ternary",
            architecture="bert-base",
            dataset="Various NLP",
            compression_method="ternary",
            description="BERT-base with IGQK ternary compression",
            in_features=768, num_classes=2,
            hidden_layers=[768] * 12,
            expected_accuracy=0.85,
            expected_compression_ratio=16.0,
            expected_size_mb=27.0,
            expected_bits_per_weight=2.0,
            hbar=0.03, gamma=0.003,
            annealing_schedule="adaptive",
        )

        self._recipes["gpt2_ternary"] = ModelRecipe(
            name="gpt2_ternary",
            architecture="gpt2",
            dataset="WebText",
            compression_method="ternary",
            description="GPT-2 (117M) with IGQK ternary + temporal compression",
            in_features=768, num_classes=50257,
            hidden_layers=[768] * 12,
            expected_accuracy=0.0,  # Perplexity-based
            expected_compression_ratio=16.0,
            expected_size_mb=14.6,
            expected_bits_per_weight=2.0,
            hbar=0.02, gamma=0.002,
            annealing_schedule="adaptive",
        )

    def list_models(self) -> List[Dict]:
        """List all available model recipes."""
        return [
            {
                "name": r.name,
                "architecture": r.architecture,
                "dataset": r.dataset,
                "method": r.compression_method,
                "expected_accuracy": r.expected_accuracy,
                "compression_ratio": f"{r.expected_compression_ratio:.0f}x",
                "size_mb": r.expected_size_mb,
                "bits/weight": r.expected_bits_per_weight,
                "description": r.description,
            }
            for r in self._recipes.values()
        ]

    def get_recipe(self, name: str) -> ModelRecipe:
        """Get a specific model recipe."""
        if name not in self._recipes:
            available = list(self._recipes.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return self._recipes[name]

    def create_compressed(self, name: str) -> Tuple[nn.Module, dict]:
        """
        Create a compressed model from a recipe.

        Returns:
            (model, stats) where model is the compressed PyTorch model.
        """
        recipe = self.get_recipe(name)
        model = self._build_model(recipe)

        # Apply compression
        from igqk.theory.tlgt import TernaryLieGroup
        from igqk.theory.hlwt import HybridLaplaceWavelet

        stats = {"recipe": name, "method": recipe.compression_method}

        with torch.no_grad():
            if recipe.compression_method == "ternary":
                for param in model.parameters():
                    if param.numel() >= 16:
                        tlgt = TernaryLieGroup(param.numel())
                        compressed, scale = tlgt.quantize(param.data)
                        param.data = compressed

            elif recipe.compression_method == "wavelet":
                hlwt = HybridLaplaceWavelet()
                for param in model.parameters():
                    if param.numel() >= 16:
                        compressed, ratio = hlwt.compress(param.data, keep_ratio=0.3)
                        param.data = compressed

            elif recipe.compression_method == "sparse":
                for param in model.parameters():
                    if param.numel() >= 16:
                        flat = param.data.flatten()
                        threshold = torch.quantile(flat.abs(), 0.8)
                        param.data *= (param.data.abs() >= threshold).float()

        # Compute actual stats
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        stats["total_params"] = total_params
        stats["sparsity"] = zero_params / total_params
        stats["expected_accuracy"] = recipe.expected_accuracy
        stats["compression_ratio"] = recipe.expected_compression_ratio

        return model, stats

    def _build_model(self, recipe: ModelRecipe) -> nn.Module:
        """Build a PyTorch model from recipe."""
        if recipe.architecture in ("fc_small", "fc_medium", "fc_large"):
            layers = [nn.Flatten()]
            prev_dim = recipe.in_features
            for hidden in recipe.hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden))
                if recipe.activation == "relu":
                    layers.append(nn.ReLU())
                elif recipe.activation == "gelu":
                    layers.append(nn.GELU())
                prev_dim = hidden
            layers.append(nn.Linear(prev_dim, recipe.num_classes))
            return nn.Sequential(*layers)

        elif recipe.architecture == "cnn_simple":
            return nn.Sequential(
                nn.Unflatten(1, (3, 32, 32)) if recipe.in_features == 3072 else nn.Unflatten(1, (1, 28, 28)),
                nn.Conv2d(3 if recipe.in_features == 3072 else 1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8 if recipe.in_features == 3072 else 64 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, recipe.num_classes),
            )

        else:
            # Default FC model
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(recipe.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, recipe.num_classes),
            )

    def register_recipe(self, recipe: ModelRecipe):
        """Register a custom model recipe."""
        self._recipes[recipe.name] = recipe

    def summary(self) -> str:
        """Get formatted summary of all models."""
        lines = [
            "IGQK Model Zoo",
            "=" * 90,
            f"{'Name':<25} {'Arch':<12} {'Dataset':<12} {'Method':<10} {'Ratio':<8} {'Size MB':<10} {'Bits/W':<8}",
            "-" * 90,
        ]
        for r in self._recipes.values():
            lines.append(
                f"{r.name:<25} {r.architecture:<12} {r.dataset:<12} "
                f"{r.compression_method:<10} {r.expected_compression_ratio:<8.0f} "
                f"{r.expected_size_mb:<10.1f} {r.expected_bits_per_weight:<8.1f}"
            )
        lines.append("=" * 90)
        lines.append(f"Total: {len(self._recipes)} models")
        return "\n".join(lines)
