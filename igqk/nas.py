"""
Compression-Aware Neural Architecture Search (Q-NAS).

Instead of designing a model and then compressing it,
search for architectures that are INHERENTLY compressible.

The search uses quantum entropy as a compressibility predictor:
architectures whose quantum state has low entropy during training
will compress well.

Search space:
    - Layer widths
    - Activation functions
    - Skip connections
    - Depth
    - Attention head counts

Fitness function:
    F(arch) = α · Accuracy(arch) + β · Compressibility(ρ_arch)
    where Compressibility = 1 - normalized_entropy(ρ)
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

from igqk.core.quantum_state import QuantumState


@dataclass
class Architecture:
    """A candidate neural network architecture."""
    id: int
    layer_widths: List[int]
    activations: List[str]
    skip_connections: List[bool]
    dropout_rates: List[float]
    fitness: float = 0.0
    accuracy: float = 0.0
    compressibility: float = 0.0
    entropy: float = 0.0
    params: int = 0


class CompressionAwareNAS:
    """
    Neural Architecture Search optimizing for both accuracy AND compressibility.

    Uses evolutionary search with quantum entropy as compressibility proxy.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        accuracy_weight: float = 0.6,
        compressibility_weight: float = 0.4,
        population_size: int = 20,
        generations: int = 10,
        min_layers: int = 2,
        max_layers: int = 6,
        quantum_rank: int = 5,
    ):
        self.in_features = in_features
        self.num_classes = num_classes
        self.alpha = accuracy_weight
        self.beta = compressibility_weight
        self.population_size = population_size
        self.generations = generations
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.quantum_rank = quantum_rank

        self._arch_counter = 0
        self._history: List[Architecture] = []
        self._best: Optional[Architecture] = None

    def search(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        verbose: bool = True,
    ) -> Architecture:
        """
        Run NAS to find optimal compressible architecture.

        Args:
            train_fn: Function(model, epochs) → trains model in-place.
            eval_fn: Function(model) → accuracy (0-1).
            verbose: Print progress.

        Returns:
            Best architecture found.
        """
        # Initialize population
        population = [self._random_architecture() for _ in range(self.population_size)]

        for gen in range(self.generations):
            if verbose:
                print(f"\n[Q-NAS] Generation {gen+1}/{self.generations}")

            # Evaluate each architecture
            for arch in population:
                if arch.fitness > 0:
                    continue  # Already evaluated

                model = self._build_model(arch)
                arch.params = sum(p.numel() for p in model.parameters())

                # Quick training
                train_fn(model, epochs=3)

                # Evaluate accuracy
                arch.accuracy = eval_fn(model)

                # Evaluate compressibility via quantum entropy
                arch.compressibility = self._evaluate_compressibility(model)
                arch.entropy = 1.0 - arch.compressibility

                # Combined fitness
                arch.fitness = self.alpha * arch.accuracy + self.beta * arch.compressibility

                self._history.append(arch)

                if verbose:
                    print(
                        f"  Arch {arch.id}: "
                        f"layers={arch.layer_widths}, "
                        f"acc={arch.accuracy:.3f}, "
                        f"comp={arch.compressibility:.3f}, "
                        f"fitness={arch.fitness:.3f}, "
                        f"params={arch.params:,}"
                    )

            # Selection (top 50%)
            population.sort(key=lambda a: a.fitness, reverse=True)
            survivors = population[:self.population_size // 2]

            # Update best
            if self._best is None or population[0].fitness > self._best.fitness:
                self._best = population[0]

            if verbose:
                print(f"  Best: fitness={self._best.fitness:.3f}, "
                      f"layers={self._best.layer_widths}")

            # Generate children via crossover and mutation
            children = []
            while len(children) < self.population_size // 2:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                children.append(child)

            population = survivors + children

        return self._best

    def _random_architecture(self) -> Architecture:
        """Generate a random architecture."""
        self._arch_counter += 1
        num_layers = random.randint(self.min_layers, self.max_layers)

        widths = []
        for i in range(num_layers):
            width = random.choice([32, 64, 128, 256, 512])
            widths.append(width)

        activations = [random.choice(["relu", "gelu", "tanh"]) for _ in range(num_layers)]
        skip = [random.random() > 0.7 for _ in range(num_layers)]
        dropout = [random.choice([0.0, 0.1, 0.2, 0.3]) for _ in range(num_layers)]

        return Architecture(
            id=self._arch_counter,
            layer_widths=widths,
            activations=activations,
            skip_connections=skip,
            dropout_rates=dropout,
        )

    def _build_model(self, arch: Architecture) -> nn.Module:
        """Build a PyTorch model from architecture specification."""
        layers = []
        prev_dim = self.in_features

        for i, width in enumerate(arch.layer_widths):
            layers.append(nn.Linear(prev_dim, width))

            act = arch.activations[i]
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "gelu":
                layers.append(nn.GELU())
            elif act == "tanh":
                layers.append(nn.Tanh())

            if arch.dropout_rates[i] > 0:
                layers.append(nn.Dropout(arch.dropout_rates[i]))

            prev_dim = width

        layers.append(nn.Linear(prev_dim, self.num_classes))

        return nn.Sequential(nn.Flatten(), *layers)

    def _evaluate_compressibility(self, model: nn.Module) -> float:
        """
        Evaluate how compressible a model is using quantum entropy.

        Low entropy → highly compressible → high score
        High entropy → hard to compress → low score
        """
        entropies = []
        for param in model.parameters():
            if param.numel() < 16:
                continue
            flat = param.detach().flatten()
            rank = min(self.quantum_rank, flat.shape[0])
            rho = QuantumState.from_point(flat, rank=rank)
            max_entropy = np.log(rank)
            normalized = rho.entropy() / max_entropy if max_entropy > 0 else 0
            entropies.append(normalized)

        if not entropies:
            return 0.5

        avg_entropy = np.mean(entropies)
        return float(np.clip(1.0 - avg_entropy, 0.0, 1.0))  # Low entropy = high compressibility

    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover two parent architectures."""
        self._arch_counter += 1

        # Take widths from one parent, activations from another
        if random.random() > 0.5:
            widths = parent1.layer_widths.copy()
            activations = parent2.activations.copy()
        else:
            widths = parent2.layer_widths.copy()
            activations = parent1.activations.copy()

        # Match lengths
        min_len = min(len(widths), len(activations))
        widths = widths[:min_len]
        activations = activations[:min_len]
        skip = [random.choice([parent1.skip_connections, parent2.skip_connections])[i % len(parent1.skip_connections)] for i in range(min_len)]
        dropout = [random.choice([parent1.dropout_rates, parent2.dropout_rates])[i % len(parent1.dropout_rates)] for i in range(min_len)]

        return Architecture(
            id=self._arch_counter,
            layer_widths=widths,
            activations=activations,
            skip_connections=skip,
            dropout_rates=dropout,
        )

    def _mutate(self, arch: Architecture, mutation_rate: float = 0.3) -> Architecture:
        """Mutate an architecture."""
        if random.random() < mutation_rate:
            # Mutate a layer width
            idx = random.randint(0, len(arch.layer_widths) - 1)
            arch.layer_widths[idx] = random.choice([32, 64, 128, 256, 512])

        if random.random() < mutation_rate:
            # Mutate activation
            idx = random.randint(0, len(arch.activations) - 1)
            arch.activations[idx] = random.choice(["relu", "gelu", "tanh"])

        if random.random() < mutation_rate * 0.5:
            # Add or remove a layer
            if len(arch.layer_widths) < self.max_layers:
                arch.layer_widths.append(random.choice([32, 64, 128, 256]))
                arch.activations.append(random.choice(["relu", "gelu", "tanh"]))
                arch.skip_connections.append(False)
                arch.dropout_rates.append(0.0)
            elif len(arch.layer_widths) > self.min_layers:
                arch.layer_widths.pop()
                arch.activations.pop()
                arch.skip_connections.pop()
                arch.dropout_rates.pop()

        return arch

    @property
    def best_architecture(self) -> Optional[Architecture]:
        return self._best

    def build_best_model(self) -> Optional[nn.Module]:
        """Build PyTorch model from best found architecture."""
        if self._best is None:
            return None
        return self._build_model(self._best)
