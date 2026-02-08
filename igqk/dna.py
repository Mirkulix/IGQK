"""
Neural Architecture DNA - Encode compression recipes as transferable "DNA".

Every model that IGQK compresses produces a unique DNA sequence that encodes:
- The architecture topology
- Per-layer compression decisions
- Optimal parameters discovered
- Quality/ratio tradeoffs

This DNA can be:
1. SHARED: Send DNA to another team, they apply it to similar models
2. MUTATED: Evolve DNA to find even better compression recipes
3. CROSSED: Combine DNA from two models to compress a hybrid
4. FINGERPRINTED: Identify if a model was compressed by IGQK

    DNA Sequence:
    ┌──────────────────────────────────────────────────┐
    │ IGQK-DNA-v1                                      │
    │ Header: arch_hash | num_layers | total_params     │
    │                                                   │
    │ Gene 0: [TERN|0.70|1.00|0.85|F] ─ layer0.weight │
    │ Gene 1: [SPRS|0.30|0.60|0.92|F] ─ layer1.weight │
    │ Gene 2: [WAVE|0.50|0.45|0.78|T] ─ layer2.weight │
    │ Gene 3: [LRNK|0.00|0.30|0.95|F] ─ layer3.weight │
    │                                                   │
    │ Fitness: 0.89 | Ratio: 12.5x | Quality: 0.97    │
    └──────────────────────────────────────────────────┘
"""

import json
import hashlib
import time
import random
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict


# Method encoding for compact DNA representation
METHOD_CODES = {
    "ternary": "TERN",
    "sparse": "SPRS",
    "wavelet": "WAVE",
    "lowrank": "LRNK",
    "binary": "BNRY",
    "adaptive_sparse": "ADSP",
    "none": "NONE",
}
CODE_TO_METHOD = {v: k for k, v in METHOD_CODES.items()}


@dataclass
class Gene:
    """A single gene encoding one layer's compression recipe."""
    layer_name: str
    layer_shape: List[int]
    method: str  # compression method
    threshold: float = 0.7       # method-specific threshold
    keep_ratio: float = 0.5      # fraction to keep
    quality_score: float = 0.0   # measured quality after compression
    use_two_stage: bool = False  # apply two-stage refinement

    def encode(self) -> str:
        """Encode gene as compact string."""
        code = METHOD_CODES.get(self.method, "NONE")
        flag = "T" if self.use_two_stage else "F"
        return f"[{code}|{self.threshold:.2f}|{self.keep_ratio:.2f}|{self.quality_score:.2f}|{flag}]"

    @classmethod
    def decode(cls, encoded: str, layer_name: str = "", layer_shape: Optional[List[int]] = None) -> "Gene":
        """Decode gene from compact string."""
        inner = encoded.strip("[]")
        parts = inner.split("|")
        method = CODE_TO_METHOD.get(parts[0], "ternary")
        return cls(
            layer_name=layer_name,
            layer_shape=layer_shape or [],
            method=method,
            threshold=float(parts[1]),
            keep_ratio=float(parts[2]),
            quality_score=float(parts[3]),
            use_two_stage=parts[4] == "T",
        )

    def mutate(self, rate: float = 0.3) -> "Gene":
        """Mutate this gene."""
        child = copy.deepcopy(self)
        if random.random() < rate:
            methods = list(METHOD_CODES.keys())
            child.method = random.choice(methods)
        if random.random() < rate:
            child.threshold = max(0.1, min(2.0, child.threshold + random.gauss(0, 0.2)))
        if random.random() < rate:
            child.keep_ratio = max(0.05, min(0.95, child.keep_ratio + random.gauss(0, 0.15)))
        if random.random() < rate:
            child.use_two_stage = not child.use_two_stage
        child.quality_score = 0.0  # Reset until re-evaluated
        return child


@dataclass
class CompressionDNA:
    """
    Full DNA sequence encoding a model's compression recipe.
    """
    # Identity
    dna_id: str = ""
    created_at: float = 0.0
    version: str = "1"

    # Architecture fingerprint
    arch_hash: str = ""
    total_params: int = 0
    num_layers: int = 0

    # Genes (per-layer recipes)
    genes: List[Gene] = field(default_factory=list)

    # Fitness metrics
    fitness: float = 0.0
    compression_ratio: float = 1.0
    quality: float = 0.0

    # Lineage
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    mutations: int = 0

    def encode(self) -> str:
        """Encode entire DNA as human-readable string."""
        lines = [
            f"IGQK-DNA-v{self.version}",
            f"ID:{self.dna_id}",
            f"ARCH:{self.arch_hash}|{self.num_layers}|{self.total_params}",
            f"GEN:{self.generation}|MUT:{self.mutations}",
        ]
        for i, gene in enumerate(self.genes):
            lines.append(f"G{i}:{gene.encode()}:{gene.layer_name}")
        lines.append(
            f"FIT:{self.fitness:.4f}|RAT:{self.compression_ratio:.2f}|"
            f"QUA:{self.quality:.4f}"
        )
        return "\n".join(lines)

    @classmethod
    def decode(cls, encoded: str) -> "CompressionDNA":
        """Decode DNA from string."""
        dna = cls()
        for line in encoded.strip().split("\n"):
            if line.startswith("IGQK-DNA-v"):
                dna.version = line.split("v")[1]
            elif line.startswith("ID:"):
                dna.dna_id = line[3:]
            elif line.startswith("ARCH:"):
                parts = line[5:].split("|")
                dna.arch_hash = parts[0]
                dna.num_layers = int(parts[1])
                dna.total_params = int(parts[2])
            elif line.startswith("GEN:"):
                gen_mut = line.split("|")
                dna.generation = int(gen_mut[0].split(":")[1])
                dna.mutations = int(gen_mut[1].split(":")[1])
            elif line.startswith("G"):
                colon_parts = line.split(":", 2)
                gene_str = colon_parts[1]
                layer_name = colon_parts[2] if len(colon_parts) > 2 else ""
                gene = Gene.decode(gene_str, layer_name=layer_name)
                dna.genes.append(gene)
            elif line.startswith("FIT:"):
                parts = line.split("|")
                dna.fitness = float(parts[0].split(":")[1])
                dna.compression_ratio = float(parts[1].split(":")[1])
                dna.quality = float(parts[2].split(":")[1])
        return dna

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "dna_id": self.dna_id,
            "created_at": self.created_at,
            "version": self.version,
            "arch_hash": self.arch_hash,
            "total_params": self.total_params,
            "num_layers": self.num_layers,
            "genes": [asdict(g) for g in self.genes],
            "fitness": self.fitness,
            "compression_ratio": self.compression_ratio,
            "quality": self.quality,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "mutations": self.mutations,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CompressionDNA":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        genes = [Gene(**g) for g in data.pop("genes", [])]
        dna = cls(**data)
        dna.genes = genes
        return dna


class DNAExtractor:
    """
    Extract compression DNA from models and apply DNA to new models.
    """

    def extract(self, model: nn.Module, compressed_model: Optional[nn.Module] = None) -> CompressionDNA:
        """
        Extract DNA from a model (optionally comparing to its compressed version).

        Args:
            model: Original or compressed model
            compressed_model: If provided, compare to infer compression recipe

        Returns:
            CompressionDNA encoding the compression recipe
        """
        dna = CompressionDNA(
            dna_id=self._generate_id(),
            created_at=time.time(),
        )

        # Architecture fingerprint
        param_shapes = []
        total_params = 0
        for name, param in model.named_parameters():
            if param.numel() < 4:
                continue
            param_shapes.append((name, list(param.shape)))
            total_params += param.numel()

        dna.arch_hash = hashlib.md5(
            str(param_shapes).encode()
        ).hexdigest()[:12]
        dna.total_params = total_params
        dna.num_layers = len(param_shapes)

        # Extract per-layer genes
        compressed_params = dict(compressed_model.named_parameters()) if compressed_model else {}

        for name, shape in param_shapes:
            param = dict(model.named_parameters())[name]
            comp_param = compressed_params.get(name)

            gene = self._infer_gene(name, param, comp_param)
            dna.genes.append(gene)

        return dna

    def _infer_gene(
        self, name: str, original: nn.Parameter,
        compressed: Optional[nn.Parameter] = None
    ) -> Gene:
        """Infer the compression gene for a layer."""
        w = original.data.flatten().float()
        shape = list(original.shape)

        if compressed is None:
            # Analyze the weights to suggest optimal compression
            method, threshold, keep_ratio = self._suggest_compression(w)
            return Gene(
                layer_name=name, layer_shape=shape,
                method=method, threshold=threshold,
                keep_ratio=keep_ratio,
            )

        # Compare original and compressed to infer what was done
        c = compressed.data.flatten().float()

        unique_vals = c.unique()
        sparsity = (c == 0).float().mean().item()

        if len(unique_vals) <= 3 and sparsity > 0.1:
            method = "ternary"
            std = w.std().item()
            threshold = 0.7  # Default
        elif sparsity > 0.5:
            method = "sparse"
            threshold = 0.5
        elif len(unique_vals) <= 2:
            method = "binary"
            threshold = 0.5
        else:
            method = "lowrank"
            threshold = 0.3

        # Measure quality
        if w.norm() > 0:
            quality = 1.0 - (w - c).norm().item() / w.norm().item()
        else:
            quality = 1.0

        keep_ratio = 1.0 - sparsity

        return Gene(
            layer_name=name, layer_shape=shape,
            method=method, threshold=threshold,
            keep_ratio=keep_ratio,
            quality_score=max(0, quality),
        )

    def _suggest_compression(self, weights: torch.Tensor) -> Tuple[str, float, float]:
        """Suggest optimal compression based on weight analysis."""
        std = weights.std().item()
        sparsity = (weights.abs() < 0.01 * max(std, 1e-8)).float().mean().item()

        # Compute kurtosis
        if std > 0:
            centered = weights - weights.mean()
            kurtosis = (centered ** 4).mean().item() / (std ** 4) - 3
        else:
            kurtosis = 0

        if sparsity > 0.4:
            return "sparse", 0.5, 1.0 - sparsity
        elif abs(kurtosis) < 1:
            return "ternary", 0.7 * std, 0.5
        elif kurtosis > 3:
            return "adaptive_sparse", 0.9, 0.3
        else:
            return "wavelet", 0.5, 0.4

    def apply(self, dna: CompressionDNA, model: nn.Module) -> nn.Module:
        """
        Apply compression DNA to a model.

        Args:
            dna: CompressionDNA to apply
            model: Model to compress

        Returns:
            Compressed model (modified in-place)
        """
        param_dict = dict(model.named_parameters())

        with torch.no_grad():
            for gene in dna.genes:
                if gene.layer_name not in param_dict:
                    continue
                if gene.method == "none":
                    continue

                param = param_dict[gene.layer_name]
                self._apply_gene(param, gene)

        return model

    def _apply_gene(self, param: nn.Parameter, gene: Gene):
        """Apply a single gene to a parameter."""
        if gene.method == "ternary":
            std = param.data.std()
            threshold = gene.threshold * std
            result = torch.zeros_like(param.data)
            result[param.data > threshold] = std
            result[param.data < -threshold] = -std
            param.data = result

        elif gene.method == "sparse":
            flat = param.data.flatten()
            k = max(1, int(gene.keep_ratio * flat.numel()))
            _, indices = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat)
            mask[indices] = 1.0
            param.data = (flat * mask).reshape(param.data.shape)

        elif gene.method == "binary":
            median = param.data.median()
            std = param.data.std()
            param.data = torch.where(param.data > median, std, -std)

        elif gene.method == "wavelet":
            flat = param.data.flatten().float()
            n = flat.numel()
            padded = flat if n % 2 == 0 else torch.cat([flat, torch.zeros(1)])
            freq = torch.fft.rfft(padded)
            k = max(1, int(gene.keep_ratio * freq.numel()))
            _, indices = torch.topk(freq.abs(), k)
            mask = torch.zeros_like(freq)
            mask[indices] = 1.0
            reconstructed = torch.fft.irfft(freq * mask, n=padded.numel())[:n]
            param.data = reconstructed.reshape(param.data.shape).to(param.data.dtype)

        elif gene.method == "lowrank":
            if param.data.dim() >= 2:
                W = param.data.float()
                shape = W.shape
                W2d = W.reshape(shape[0], -1)
                U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
                rank = max(1, int(gene.keep_ratio * min(W2d.shape)))
                param.data = (
                    U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
                ).reshape(shape).to(param.data.dtype)

        elif gene.method == "adaptive_sparse":
            threshold = torch.quantile(
                param.data.abs().flatten().float(), gene.threshold
            )
            param.data *= (param.data.abs() >= threshold).float()

        # Two-stage post-processing
        if gene.use_two_stage:
            std = param.data.std()
            if std > 0:
                threshold = 0.5 * std
                stage2 = torch.zeros_like(param.data)
                stage2[param.data > threshold] = std
                stage2[param.data < -threshold] = -std
                param.data = 0.5 * param.data + 0.5 * stage2

    def crossover(self, dna_a: CompressionDNA, dna_b: CompressionDNA) -> CompressionDNA:
        """
        Cross two DNA sequences to create a child.

        Takes genes from both parents, preferring higher quality genes.
        """
        child = CompressionDNA(
            dna_id=self._generate_id(),
            created_at=time.time(),
            arch_hash=dna_a.arch_hash,
            total_params=dna_a.total_params,
            num_layers=dna_a.num_layers,
            parent_ids=[dna_a.dna_id, dna_b.dna_id],
            generation=max(dna_a.generation, dna_b.generation) + 1,
        )

        # For each gene position, pick the better parent gene
        genes_b = {g.layer_name: g for g in dna_b.genes}
        for gene_a in dna_a.genes:
            gene_b = genes_b.get(gene_a.layer_name)
            if gene_b and gene_b.quality_score > gene_a.quality_score:
                child.genes.append(copy.deepcopy(gene_b))
            else:
                child.genes.append(copy.deepcopy(gene_a))

        return child

    def mutate(self, dna: CompressionDNA, rate: float = 0.2) -> CompressionDNA:
        """Mutate a DNA sequence."""
        mutant = CompressionDNA(
            dna_id=self._generate_id(),
            created_at=time.time(),
            arch_hash=dna.arch_hash,
            total_params=dna.total_params,
            num_layers=dna.num_layers,
            parent_ids=[dna.dna_id],
            generation=dna.generation + 1,
            mutations=dna.mutations + 1,
        )
        mutant.genes = [g.mutate(rate) for g in dna.genes]
        return mutant

    def fingerprint(self, model: nn.Module) -> str:
        """
        Generate a compression fingerprint for a model.
        Can identify if and how a model was compressed.
        """
        indicators = []
        for name, param in model.named_parameters():
            w = param.data.flatten().float()
            unique = w.unique().numel()
            sparsity = (w == 0).float().mean().item()
            indicators.append(f"{unique}:{sparsity:.2f}")

        raw = "|".join(indicators)
        return "IGQK-FP-" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _generate_id(self) -> str:
        raw = f"{time.time()}-{random.random()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
