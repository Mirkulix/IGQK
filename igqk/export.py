"""
ONNX / TensorRT Export for IGQK compressed models.

Export compressed models to deployment-ready formats:
- ONNX: Universal interchange format
- ONNX with optimization: Graph-level optimizations (constant folding, fusion)
- TensorRT: NVIDIA GPU-optimized inference (if available)

Usage:
    from igqk.export import ONNXExporter
    exporter = ONNXExporter(model, input_shape=(1, 784))
    exporter.export("model.onnx")
    exporter.export("model_optimized.onnx", optimize=True)

CLI:
    python -m igqk.export --checkpoint model.pt --output model.onnx
"""

import torch
import torch.nn as nn
import os
import time
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path


class ONNXExporter:
    """Export PyTorch models to ONNX format with optional optimization."""

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 784),
        opset_version: int = 17,
        dynamic_batch: bool = True,
    ):
        """
        Args:
            model: PyTorch model to export.
            input_shape: Example input shape (batch_size, *feature_dims).
            opset_version: ONNX opset version.
            dynamic_batch: Allow dynamic batch size in exported model.
        """
        self.model = model
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch
        self._export_info: Dict[str, Any] = {}

    def export(
        self,
        output_path: str,
        optimize: bool = False,
        quantize_onnx: bool = False,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path: Path for the .onnx file.
            optimize: Apply ONNX graph optimizations.
            quantize_onnx: Apply ONNX quantization (int8).

        Returns:
            Path to the exported file.
        """
        self.model.eval()
        dummy_input = torch.randn(*self.input_shape)

        # Dynamic axes for batch dimension
        dynamic_axes = {}
        if self.dynamic_batch:
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        start = time.time()
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes if self.dynamic_batch else None,
        )
        export_time = time.time() - start

        file_size = os.path.getsize(output_path)
        self._export_info = {
            "format": "onnx",
            "path": output_path,
            "size_mb": file_size / 1e6,
            "opset_version": self.opset_version,
            "dynamic_batch": self.dynamic_batch,
            "export_time_s": export_time,
        }

        # Optimize
        if optimize:
            output_path = self._optimize_onnx(output_path)

        # Quantize
        if quantize_onnx:
            output_path = self._quantize_onnx(output_path)

        return output_path

    def _optimize_onnx(self, model_path: str) -> str:
        """Apply ONNX graph optimizations."""
        try:
            import onnx
            from onnx import optimizer as onnx_optimizer
        except ImportError:
            try:
                import onnx
                # Use onnxoptimizer if available
                import onnxoptimizer
                model = onnx.load(model_path)
                optimized = onnxoptimizer.optimize(model)
                opt_path = model_path.replace(".onnx", "_optimized.onnx")
                onnx.save(optimized, opt_path)
                self._export_info["optimized_path"] = opt_path
                self._export_info["optimized_size_mb"] = os.path.getsize(opt_path) / 1e6
                return opt_path
            except ImportError:
                # Fallback: use onnxruntime optimization
                try:
                    import onnxruntime as ort
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    opt_path = model_path.replace(".onnx", "_optimized.onnx")
                    sess_options.optimized_model_filepath = opt_path
                    ort.InferenceSession(model_path, sess_options)
                    if os.path.exists(opt_path):
                        self._export_info["optimized_path"] = opt_path
                        self._export_info["optimized_size_mb"] = os.path.getsize(opt_path) / 1e6
                        return opt_path
                except ImportError:
                    pass

        return model_path

    def _quantize_onnx(self, model_path: str) -> str:
        """Apply ONNX quantization (int8)."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quant_path = model_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(model_path, quant_path, weight_type=QuantType.QInt8)
            self._export_info["quantized_path"] = quant_path
            self._export_info["quantized_size_mb"] = os.path.getsize(quant_path) / 1e6
            return quant_path
        except ImportError:
            return model_path

    def export_tensorrt(
        self,
        onnx_path: str,
        output_path: str,
        fp16: bool = True,
        max_batch_size: int = 32,
    ) -> Optional[str]:
        """
        Convert ONNX model to TensorRT engine.

        Requires: tensorrt, pycuda

        Args:
            onnx_path: Path to ONNX model.
            output_path: Path for TensorRT engine.
            fp16: Enable FP16 precision.
            max_batch_size: Maximum batch size for optimization.

        Returns:
            Path to TensorRT engine or None if TensorRT not available.
        """
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"TensorRT parse error: {parser.get_error(i)}")
                    return None

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            engine = builder.build_serialized_network(network, config)
            if engine is None:
                return None

            with open(output_path, "wb") as f:
                f.write(engine)

            self._export_info["tensorrt_path"] = output_path
            self._export_info["tensorrt_size_mb"] = os.path.getsize(output_path) / 1e6
            self._export_info["tensorrt_fp16"] = fp16
            return output_path

        except ImportError:
            return None

    def validate(self, onnx_path: str, num_samples: int = 10, atol: float = 1e-4) -> dict:
        """
        Validate ONNX model outputs match PyTorch.

        Args:
            onnx_path: Path to ONNX model.
            num_samples: Number of random samples to validate.
            atol: Absolute tolerance for comparison.

        Returns:
            Validation results dict.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            return {"valid": False, "error": "onnxruntime not installed"}

        self.model.eval()
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        errors = []
        for _ in range(num_samples):
            x = torch.randn(*self.input_shape)

            # PyTorch output
            with torch.no_grad():
                pt_out = self.model(x).numpy()

            # ONNX output
            ort_out = session.run(None, {input_name: x.numpy()})[0]

            max_diff = abs(pt_out - ort_out).max()
            errors.append(max_diff)

        max_error = max(errors)
        avg_error = sum(errors) / len(errors)

        return {
            "valid": max_error < atol,
            "max_error": float(max_error),
            "avg_error": float(avg_error),
            "num_samples": num_samples,
            "tolerance": atol,
        }

    def summary(self) -> str:
        """Get export summary."""
        lines = [
            "IGQK Model Export Summary",
            "=" * 40,
        ]
        for key, val in self._export_info.items():
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"  {key}: {val}")

        # Model stats
        total_params = sum(p.numel() for p in self.model.parameters())
        zero_params = sum((p == 0).sum().item() for p in self.model.parameters())
        lines.append(f"  total_params: {total_params:,}")
        lines.append(f"  zero_params: {zero_params:,} ({100*zero_params/total_params:.1f}%)")

        return "\n".join(lines)


def main():
    """CLI entry point for ONNX export."""
    import argparse

    parser = argparse.ArgumentParser(description="IGQK ONNX Export")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output", default="model.onnx", help="Output ONNX path")
    parser.add_argument("--optimize", action="store_true", help="Apply optimizations")
    parser.add_argument("--quantize", action="store_true", help="Quantize to int8")
    parser.add_argument("--input-shape", default="1,784", help="Input shape (comma-separated)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--validate", action="store_true", help="Validate output")
    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Infer architecture from state dict
    keys = list(state_dict.keys())
    first_weight = state_dict[keys[0]]
    in_features = first_weight.shape[1] if first_weight.dim() == 2 else first_weight.shape[0]
    last_weight_key = [k for k in keys if "weight" in k][-1]
    num_classes = state_dict[last_weight_key].shape[0]

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )
    model.load_state_dict(state_dict, strict=False)

    # Export
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    exporter = ONNXExporter(model, input_shape=input_shape, opset_version=args.opset)
    path = exporter.export(args.output, optimize=args.optimize, quantize_onnx=args.quantize)

    print(exporter.summary())

    if args.validate:
        result = exporter.validate(path)
        print(f"\nValidation: {'PASSED' if result['valid'] else 'FAILED'}")
        print(f"  Max error: {result['max_error']:.6f}")


if __name__ == "__main__":
    main()
