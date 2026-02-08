"""
Self-Evolving Strategy Engine - Compression strategies that mutate and evolve.

Unlike static compression, IGQK strategies EVOLVE over time:
1. Start with base strategies (ternary, wavelet, sparse)
2. Mutate parameters (thresholds, ratios, combinations)
3. Evaluate fitness (compression quality)
4. Select best strategies via tournament selection
5. Cross strategies to create NEW methods nobody programmed

This is Artificial Evolution applied to compression algorithms.
The system INVENTS new compression techniques autonomously.

    Generation 0: [ternary, wavelet, sparse]
    Generation 1: [ternary_v2, wavelet_aggressive, sparse_adaptive]
    Generation 5: [hybrid_ternary_wavelet, quantum_sparse_v3, ...]
    Generation N: [methods that don't exist yet...]
"""

import torch
import torch.nn as nn
import copy
import random
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class Strategy:
    """A compression strategy with evolvable parameters."""
    id: int
    name: str
    generation: int = 0

    # Evolvable parameters (genes)
    ternary_threshold: float = 0.7  # Threshold multiplier for ternary
    sparse_keep_ratio: float = 0.3  # Fraction of weights to keep
    wavelet_keep_ratio: float = 0.3
    lowrank_ratio: float = 0.3  # Fraction of singular values to keep
    blend_ternary: float = 0.0  # How much ternary in hybrid (0-1)
    blend_sparse: float = 0.0
    blend_wavelet: float = 0.0
    post_scale: float = 1.0  # Scale factor after compression
    use_two_stage: bool = False  # Apply two methods sequentially

    # Fitness (evaluated)
    fitness: float = 0.0
    compression_ratio: float = 1.0
    distortion: float = float("inf")
    evaluations: int = 0

    def mutate(self, rate: float = 0.3) -> "Strategy":
        """Create a mutated copy."""
        child = copy.deepcopy(self)
        child.generation += 1
        child.fitness = 0.0
        child.evaluations = 0

        if random.random() < rate:
            child.ternary_threshold *= random.uniform(0.5, 1.5)
            child.ternary_threshold = max(0.1, min(2.0, child.ternary_threshold))

        if random.random() < rate:
            child.sparse_keep_ratio = random.uniform(0.05, 0.8)

        if random.random() < rate:
            child.wavelet_keep_ratio = random.uniform(0.1, 0.9)

        if random.random() < rate:
            child.lowrank_ratio = random.uniform(0.1, 0.8)

        if random.random() < rate:
            child.blend_ternary = random.uniform(0, 1)
            child.blend_sparse = random.uniform(0, 1 - child.blend_ternary)
            child.blend_wavelet = 1.0 - child.blend_ternary - child.blend_sparse

        if random.random() < rate * 0.5:
            child.use_two_stage = not child.use_two_stage

        if random.random() < rate:
            child.post_scale = random.uniform(0.8, 1.2)

        return child

    @staticmethod
    def crossover(a: "Strategy", b: "Strategy", child_id: int) -> "Strategy":
        """Create child from two parent strategies."""
        child = Strategy(
            id=child_id,
            name=f"evolved_{child_id}",
            generation=max(a.generation, b.generation) + 1,
            ternary_threshold=random.choice([a.ternary_threshold, b.ternary_threshold]),
            sparse_keep_ratio=random.choice([a.sparse_keep_ratio, b.sparse_keep_ratio]),
            wavelet_keep_ratio=random.choice([a.wavelet_keep_ratio, b.wavelet_keep_ratio]),
            lowrank_ratio=random.choice([a.lowrank_ratio, b.lowrank_ratio]),
            blend_ternary=(a.blend_ternary + b.blend_ternary) / 2,
            blend_sparse=(a.blend_sparse + b.blend_sparse) / 2,
            blend_wavelet=(a.blend_wavelet + b.blend_wavelet) / 2,
            use_two_stage=random.choice([a.use_two_stage, b.use_two_stage]),
            post_scale=(a.post_scale + b.post_scale) / 2,
        )
        return child


class EvolutionEngine:
    """
    Evolves compression strategies through artificial evolution.

    Strategies are evaluated on actual models, and the best strategies
    survive and reproduce, creating new compression techniques.
    """

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.3,
        elite_ratio: float = 0.2,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self._id_counter = 0
        self.population: List[Strategy] = []
        self.hall_of_fame: List[Strategy] = []
        self._initialize_population()

    def _initialize_population(self):
        """Create initial population with known good strategies."""
        base_strategies = [
            {"name": "ternary_standard", "ternary_threshold": 0.7,
             "blend_ternary": 1.0, "blend_sparse": 0.0, "blend_wavelet": 0.0},
            {"name": "sparse_aggressive", "sparse_keep_ratio": 0.1,
             "blend_ternary": 0.0, "blend_sparse": 1.0, "blend_wavelet": 0.0},
            {"name": "sparse_mild", "sparse_keep_ratio": 0.5,
             "blend_ternary": 0.0, "blend_sparse": 1.0, "blend_wavelet": 0.0},
            {"name": "wavelet_standard", "wavelet_keep_ratio": 0.3,
             "blend_ternary": 0.0, "blend_sparse": 0.0, "blend_wavelet": 1.0},
            {"name": "hybrid_ternary_sparse",
             "blend_ternary": 0.5, "blend_sparse": 0.5, "blend_wavelet": 0.0},
        ]

        for base in base_strategies:
            self._id_counter += 1
            s = Strategy(id=self._id_counter, **base)
            self.population.append(s)

        # Fill rest with mutations
        while len(self.population) < self.population_size:
            parent = random.choice(self.population[:len(base_strategies)])
            self._id_counter += 1
            child = parent.mutate(self.mutation_rate)
            child.id = self._id_counter
            child.name = f"mutant_{self._id_counter}"
            self.population.append(child)

    def evaluate_strategy(
        self,
        strategy: Strategy,
        test_weights: torch.Tensor,
    ) -> float:
        """Evaluate a strategy on test weights."""
        compressed = self._apply_strategy(strategy, test_weights)

        distortion = (test_weights - compressed).norm().item()
        original_norm = test_weights.norm().item() + 1e-10
        relative_distortion = distortion / original_norm

        # Compression ratio estimate
        unique_vals = compressed.flatten().unique().numel()
        zeros = (compressed == 0).float().mean().item()
        if unique_vals <= 4:
            ratio = 16.0
        elif zeros > 0.5:
            ratio = 1.0 / max(1.0 - zeros, 0.01)
        else:
            ratio = 1.0

        # Fitness = compression quality (high ratio, low distortion)
        fitness = ratio / (1.0 + 10 * relative_distortion)

        strategy.fitness = fitness
        strategy.compression_ratio = ratio
        strategy.distortion = relative_distortion
        strategy.evaluations += 1

        return fitness

    def _apply_strategy(self, strategy: Strategy, weights: torch.Tensor) -> torch.Tensor:
        """Apply a strategy to weights."""
        result = torch.zeros_like(weights)

        if strategy.blend_ternary > 0.01:
            std = weights.std()
            threshold = strategy.ternary_threshold * std
            ternary = torch.zeros_like(weights)
            ternary[weights > threshold] = std
            ternary[weights < -threshold] = -std
            result += strategy.blend_ternary * ternary

        if strategy.blend_sparse > 0.01:
            flat = weights.flatten()
            k = max(1, int(strategy.sparse_keep_ratio * flat.numel()))
            _, indices = torch.topk(flat.abs(), k)
            sparse_mask = torch.zeros_like(flat)
            sparse_mask[indices] = 1.0
            sparse = (flat * sparse_mask).reshape(weights.shape)
            result += strategy.blend_sparse * sparse

        if strategy.blend_wavelet > 0.01:
            # Simplified wavelet-like: keep largest Fourier coefficients
            flat = weights.flatten().float()
            n = flat.numel()
            # Ensure even length for rfft
            if n % 2 != 0:
                flat = torch.cat([flat, torch.zeros(1)])
            freq = torch.fft.rfft(flat)
            k = max(1, int(strategy.wavelet_keep_ratio * freq.numel()))
            _, indices = torch.topk(freq.abs(), k)
            mask = torch.zeros_like(freq)
            mask[indices] = 1.0
            reconstructed = torch.fft.irfft(freq * mask, n=flat.numel())[:n]
            result += strategy.blend_wavelet * reconstructed.reshape(weights.shape).to(weights.dtype)

        # Two-stage: apply ternary on top
        if strategy.use_two_stage:
            std = result.std()
            if std > 0:
                threshold = 0.5 * std
                stage2 = torch.zeros_like(result)
                stage2[result > threshold] = std
                stage2[result < -threshold] = -std
                result = 0.5 * result + 0.5 * stage2

        result *= strategy.post_scale
        return result

    def evolve(
        self,
        test_weights: torch.Tensor,
        verbose: bool = False,
    ) -> Strategy:
        """
        Run evolution to discover optimal compression strategy.

        Returns the best strategy found.
        """
        for gen in range(self.generations):
            # Evaluate all strategies
            for strategy in self.population:
                self.evaluate_strategy(strategy, test_weights)

            # Sort by fitness
            self.population.sort(key=lambda s: s.fitness, reverse=True)

            best = self.population[0]
            if verbose:
                avg_fitness = np.mean([s.fitness for s in self.population])
                print(
                    f"  Gen {gen}: best={best.fitness:.4f} "
                    f"(ratio={best.compression_ratio:.1f}x, "
                    f"dist={best.distortion:.4f}), "
                    f"avg={avg_fitness:.4f}"
                )

            # Update hall of fame
            if not self.hall_of_fame or best.fitness > self.hall_of_fame[0].fitness:
                self.hall_of_fame.insert(0, copy.deepcopy(best))
                self.hall_of_fame = self.hall_of_fame[:5]

            # Selection: keep elite
            elite_n = max(2, int(self.elite_ratio * self.population_size))
            elite = self.population[:elite_n]

            # Create next generation
            new_pop = list(elite)

            while len(new_pop) < self.population_size:
                if random.random() < 0.5 and len(elite) >= 2:
                    # Crossover
                    p1, p2 = random.sample(elite, 2)
                    self._id_counter += 1
                    child = Strategy.crossover(p1, p2, self._id_counter)
                    child = child.mutate(self.mutation_rate)
                else:
                    # Mutation
                    parent = random.choice(elite)
                    self._id_counter += 1
                    child = parent.mutate(self.mutation_rate)
                    child.id = self._id_counter
                    child.name = f"evolved_{self._id_counter}"
                new_pop.append(child)

            self.population = new_pop

        # Return best ever
        return self.hall_of_fame[0] if self.hall_of_fame else self.population[0]

    def best_strategy(self) -> Optional[Strategy]:
        """Get the best strategy found so far."""
        return self.hall_of_fame[0] if self.hall_of_fame else None

    def summary(self) -> str:
        """Get evolution summary."""
        lines = [
            "IGQK Self-Evolving Strategy Engine",
            "=" * 50,
            f"  Population: {self.population_size}",
            f"  Generations: {self.generations}",
            f"  Hall of Fame:",
        ]
        for i, s in enumerate(self.hall_of_fame[:5]):
            lines.append(
                f"    #{i+1}: {s.name} gen={s.generation} "
                f"fitness={s.fitness:.4f} ratio={s.compression_ratio:.1f}x "
                f"dist={s.distortion:.4f}"
            )
            lines.append(
                f"         blend: T={s.blend_ternary:.2f} "
                f"S={s.blend_sparse:.2f} W={s.blend_wavelet:.2f} "
                f"two_stage={s.use_two_stage}"
            )
        return "\n".join(lines)
