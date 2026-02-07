"""
.igqk Ultra-Compact Binary Model Format.

A new model file format specifically designed for quantum-compressed neural networks.

Standard formats waste space:
  - .pt (PyTorch):     32 bits per weight (float32)
  - .safetensors:      32 bits per weight
  - GGUF:              4-8 bits per weight (quantized)

IGQK format:
  - Ternary weights:   2 bits per weight (4 values: -1, 0, +1, scale)
  - Sparse weights:    index + value (variable)
  - Wavelet:           coefficients only (compressed)
  - Mixed:             per-layer optimal encoding

File structure:
  ┌──────────────────────────────┐
  │ Magic: "IGQK" (4 bytes)     │
  │ Version: uint16              │
  │ Flags: uint16                │
  │ Num layers: uint32           │
  │ Metadata length: uint32      │
  │ Metadata (JSON)              │
  ├──────────────────────────────┤
  │ Layer 0 header               │
  │   Name length + name         │
  │   Shape (uint32 × ndim)      │
  │   Encoding type (uint8)      │
  │   Scale factor (float32)     │
  │   Data length (uint64)       │
  │   Compressed data            │
  ├──────────────────────────────┤
  │ Layer 1 header + data        │
  │ ...                          │
  └──────────────────────────────┘
"""

import struct
import json
import io
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path

IGQK_MAGIC = b"IGQK"
IGQK_VERSION = 1

# Encoding types
ENCODING_FLOAT32 = 0
ENCODING_TERNARY = 1       # 2 bits: {-1, 0, +1} + scale
ENCODING_SPARSE = 2         # index + value pairs
ENCODING_WAVELET = 3        # wavelet coefficients
ENCODING_BINARY = 4         # 1 bit: {-1, +1} + scale
ENCODING_INT8 = 5           # 8-bit quantized


class IGQKFormat:
    """
    Save and load models in ultra-compact .igqk format.

    Achieves 16-32x smaller files than standard .pt format
    for ternary-compressed models.
    """

    @staticmethod
    def save(
        state_dict: Dict[str, torch.Tensor],
        path: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Save model in .igqk format.

        Auto-detects the best encoding for each layer.

        Args:
            state_dict: Model state dict.
            path: Output file path.
            metadata: Optional metadata (architecture, training info, etc.).

        Returns:
            File size in bytes.
        """
        if metadata is None:
            metadata = {}

        # Add automatic metadata
        metadata["format"] = "igqk"
        metadata["version"] = IGQK_VERSION
        metadata["num_layers"] = len(state_dict)
        metadata["total_params"] = sum(t.numel() for t in state_dict.values())

        meta_bytes = json.dumps(metadata).encode("utf-8")

        with open(path, "wb") as f:
            # Header
            f.write(IGQK_MAGIC)
            f.write(struct.pack("<HH", IGQK_VERSION, 0))  # version, flags
            f.write(struct.pack("<I", len(state_dict)))     # num layers
            f.write(struct.pack("<I", len(meta_bytes)))     # metadata length
            f.write(meta_bytes)

            total_original = 0
            total_compressed = 0

            for name, tensor in state_dict.items():
                tensor = tensor.detach().cpu()
                encoding, data, scale = IGQKFormat._encode_tensor(tensor)

                # Layer header
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)

                # Shape
                shape = tensor.shape
                f.write(struct.pack("<B", len(shape)))
                for dim in shape:
                    f.write(struct.pack("<I", dim))

                # Encoding info
                f.write(struct.pack("<B", encoding))
                f.write(struct.pack("<f", scale))
                f.write(struct.pack("<Q", len(data)))
                f.write(data)

                total_original += tensor.numel() * 4
                total_compressed += len(data)

            metadata["compression_ratio"] = total_compressed / total_original
            metadata["original_bytes"] = total_original
            metadata["compressed_bytes"] = total_compressed

        file_size = Path(path).stat().st_size
        return file_size

    @staticmethod
    def load(path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
        """
        Load model from .igqk format.

        Returns:
            (state_dict, metadata)
        """
        with open(path, "rb") as f:
            # Header
            magic = f.read(4)
            if magic != IGQK_MAGIC:
                raise ValueError(f"Not an IGQK file (magic: {magic})")

            version, flags = struct.unpack("<HH", f.read(4))
            num_layers = struct.unpack("<I", f.read(4))[0]
            meta_length = struct.unpack("<I", f.read(4))[0]
            metadata = json.loads(f.read(meta_length).decode("utf-8"))

            state_dict = {}

            for _ in range(num_layers):
                # Layer header
                name_length = struct.unpack("<H", f.read(2))[0]
                name = f.read(name_length).decode("utf-8")

                ndim = struct.unpack("<B", f.read(1))[0]
                shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))

                encoding = struct.unpack("<B", f.read(1))[0]
                scale = struct.unpack("<f", f.read(4))[0]
                data_length = struct.unpack("<Q", f.read(8))[0]
                data = f.read(data_length)

                tensor = IGQKFormat._decode_tensor(data, shape, encoding, scale)
                state_dict[name] = tensor

        return state_dict, metadata

    @staticmethod
    def _encode_tensor(tensor: torch.Tensor) -> Tuple[int, bytes, float]:
        """Auto-detect best encoding and encode tensor."""
        flat = tensor.flatten().float()

        # Check if ternary
        unique_vals = flat.unique()
        if len(unique_vals) <= 4:
            # Check if values are {-s, 0, +s} pattern
            nonzero = unique_vals[unique_vals != 0]
            if len(nonzero) <= 2:
                scale = nonzero.abs().mean().item() if len(nonzero) > 0 else 1.0
                return ENCODING_TERNARY, IGQKFormat._encode_ternary(flat, scale), scale

        # Check if very sparse
        sparsity = (flat == 0).float().mean().item()
        if sparsity > 0.7:
            return ENCODING_SPARSE, IGQKFormat._encode_sparse(flat), 1.0

        # Default: float32
        return ENCODING_FLOAT32, flat.detach().cpu().numpy().tobytes(), 1.0

    @staticmethod
    def _encode_ternary(flat: torch.Tensor, scale: float) -> bytes:
        """
        Encode ternary weights in 2 bits per weight.

        Encoding: 00 = 0, 01 = +scale, 10 = -scale, 11 = reserved
        """
        normalized = flat / scale if scale != 0 else flat
        # Map: -1 -> 2, 0 -> 0, +1 -> 1
        codes = torch.zeros(len(flat), dtype=torch.uint8)
        codes[normalized > 0.5] = 1
        codes[normalized < -0.5] = 2

        # Pack 4 values per byte (2 bits each)
        num_bytes = (len(codes) + 3) // 4
        packed = bytearray(num_bytes)
        for i in range(len(codes)):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            packed[byte_idx] |= int(codes[i].item()) << bit_offset

        return bytes(packed)

    @staticmethod
    def _encode_sparse(flat: torch.Tensor) -> bytes:
        """Encode sparse tensor as (index, value) pairs."""
        nonzero_mask = flat != 0
        indices = torch.where(nonzero_mask)[0].int()
        values = flat[nonzero_mask].float()

        buf = io.BytesIO()
        buf.write(struct.pack("<I", len(indices)))
        buf.write(indices.detach().cpu().numpy().tobytes())
        buf.write(values.detach().cpu().numpy().tobytes())
        return buf.getvalue()

    @staticmethod
    def _decode_tensor(
        data: bytes, shape: tuple, encoding: int, scale: float
    ) -> torch.Tensor:
        """Decode tensor from binary data."""
        numel = 1
        for d in shape:
            numel *= d

        if encoding == ENCODING_FLOAT32:
            flat = torch.from_numpy(
                np.frombuffer(data, dtype=np.float32).copy()
            )

        elif encoding == ENCODING_TERNARY:
            # Unpack 2-bit codes
            codes = torch.zeros(numel, dtype=torch.float32)
            for i in range(numel):
                byte_idx = i // 4
                bit_offset = (i % 4) * 2
                code = (data[byte_idx] >> bit_offset) & 0x03
                if code == 1:
                    codes[i] = scale
                elif code == 2:
                    codes[i] = -scale
            flat = codes

        elif encoding == ENCODING_SPARSE:
            buf = io.BytesIO(data)
            num_nonzero = struct.unpack("<I", buf.read(4))[0]
            indices = torch.from_numpy(
                np.frombuffer(buf.read(num_nonzero * 4), dtype=np.int32).copy()
            )
            values = torch.from_numpy(
                np.frombuffer(buf.read(num_nonzero * 4), dtype=np.float32).copy()
            )
            flat = torch.zeros(numel)
            flat[indices.long()] = values

        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        return flat.reshape(shape)

    @staticmethod
    def info(path: str) -> dict:
        """Get information about an .igqk file without loading weights."""
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != IGQK_MAGIC:
                raise ValueError("Not an IGQK file")

            version, flags = struct.unpack("<HH", f.read(4))
            num_layers = struct.unpack("<I", f.read(4))[0]
            meta_length = struct.unpack("<I", f.read(4))[0]
            metadata = json.loads(f.read(meta_length).decode("utf-8"))

        file_size = Path(path).stat().st_size
        metadata["file_size_bytes"] = file_size
        metadata["file_size_mb"] = file_size / (1024 * 1024)

        return metadata
