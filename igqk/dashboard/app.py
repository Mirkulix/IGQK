"""
IGQK Production Dashboard (Gradio).

Interactive web interface for:
- Model compression with live progress
- Training visualization
- Weight analysis and quantum state monitoring
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np


def create_dashboard():
    import gradio as gr

    # --- Compression Tab ---
    def compress_weights(file_obj, method, keep_ratio):
        if file_obj is None:
            return "Please upload a model checkpoint.", None, None

        from igqk.theory.tlgt import TernaryLieGroup
        from igqk.theory.hlwt import HybridLaplaceWavelet
        from igqk.compression.projection import OptimalProjection

        try:
            checkpoint = torch.load(file_obj.name, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        except Exception as e:
            return f"Error loading model: {e}", None, None

        results = []
        compressed_state = {}
        total_orig = 0
        total_comp = 0

        for name, param in state_dict.items():
            orig_size = param.numel()
            total_orig += orig_size

            if method == "Ternary (TLGT)":
                tlgt = TernaryLieGroup(param.numel())
                compressed, scale = tlgt.quantize(param)
                stats = tlgt.compression_stats(param, compressed)
                compressed_state[name] = compressed
                comp_size = int(orig_size * stats["compression_ratio"])
            elif method == "Wavelet (HLWT)":
                hlwt = HybridLaplaceWavelet()
                compressed, ratio = hlwt.compress(param, keep_ratio=keep_ratio)
                compressed_state[name] = compressed
                comp_size = int(orig_size * ratio)
            elif method == "Sparse":
                proj = OptimalProjection(submanifold_type="sparse", sparsity=keep_ratio)
                compressed = proj.projector.project(param.flatten()).view_as(param)
                compressed_state[name] = compressed
                comp_size = int(orig_size * keep_ratio)
            else:
                compressed_state[name] = param
                comp_size = orig_size

            total_comp += comp_size
            results.append(f"  {name}: {orig_size:,} -> ~{comp_size:,} params")

        ratio = total_comp / total_orig if total_orig > 0 else 1.0
        summary = (
            f"Compression: {method}\n"
            f"Original: {total_orig:,} parameters\n"
            f"Compressed: ~{total_comp:,} effective parameters\n"
            f"Ratio: {ratio:.4f} ({1/ratio:.1f}x compression)\n\n"
            + "\n".join(results)
        )

        # Save compressed model
        output_path = "/tmp/igqk_compressed.pt"
        torch.save({"model_state_dict": compressed_state, "method": method}, output_path)

        # Create histogram
        all_weights = torch.cat([p.flatten() for p in compressed_state.values()])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.hist(all_weights.numpy(), bins=100, edgecolor="black", alpha=0.7, color="#2196F3")
        ax.set_title("Compressed Weight Distribution")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Count")
        fig.tight_layout()

        return summary, fig, output_path

    # --- Analysis Tab ---
    def analyze_model(file_obj):
        if file_obj is None:
            return "Please upload a model.", None

        try:
            checkpoint = torch.load(file_obj.name, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        except Exception as e:
            return f"Error: {e}", None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        info_lines = []
        total_params = 0
        total_zeros = 0

        all_weights = []
        layer_names = []
        layer_means = []
        layer_stds = []

        for name, param in state_dict.items():
            n = param.numel()
            total_params += n
            zeros = (param == 0).sum().item()
            total_zeros += zeros
            all_weights.append(param.flatten())
            layer_names.append(name[:20])
            layer_means.append(param.float().mean().item())
            layer_stds.append(param.float().std().item())

            info_lines.append(
                f"  {name}: shape={list(param.shape)}, "
                f"mean={param.float().mean():.4f}, std={param.float().std():.4f}, "
                f"zeros={zeros}/{n} ({100*zeros/n:.1f}%)"
            )

        summary = (
            f"Total parameters: {total_params:,}\n"
            f"Zero parameters: {total_zeros:,} ({100*total_zeros/total_params:.1f}%)\n"
            f"Layers: {len(state_dict)}\n\n"
            + "\n".join(info_lines)
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Weight distribution
        cat_weights = torch.cat(all_weights).numpy()
        axes[0].hist(cat_weights, bins=100, edgecolor="black", alpha=0.7, color="#4CAF50")
        axes[0].set_title("Overall Weight Distribution")
        axes[0].set_xlabel("Value")

        # Per-layer statistics
        x = np.arange(len(layer_names))
        axes[1].bar(x, layer_stds, color="#FF9800", alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
        axes[1].set_title("Per-Layer Std Dev")
        axes[1].set_ylabel("Std")

        fig.tight_layout()
        return summary, fig

    # --- Quick Demo Tab ---
    def run_demo(compression_method, hbar, gamma):
        from igqk.core.quantum_state import QuantumState
        from igqk.core.measurement import MeasurementOperator
        from igqk.theory.tlgt import TernaryLieGroup

        dim = 100
        weights = torch.randn(dim) * 0.5

        rho = QuantumState.from_point(weights, rank=5)
        measurement = MeasurementOperator()
        discrete = measurement.measure(rho, method="optimal")

        tlgt = TernaryLieGroup(dim)
        ternary, scale = tlgt.quantize(weights)
        stats = tlgt.compression_stats(weights, ternary)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(weights.numpy(), bins=30, alpha=0.7, color="#2196F3", label="Original")
        axes[0].set_title("Original Weights")

        axes[1].hist(discrete.numpy(), bins=30, alpha=0.7, color="#4CAF50", label="Quantum")
        axes[1].set_title("Quantum Measured")

        axes[2].hist(ternary.numpy(), bins=30, alpha=0.7, color="#FF9800", label="Ternary")
        axes[2].set_title("Ternary (TLGT)")

        for ax in axes:
            ax.set_xlabel("Weight Value")
            ax.legend()
        fig.tight_layout()

        result = (
            f"Quantum State: dim={rho.dim}, rank={rho.rank}\n"
            f"  Entropy: {rho.entropy():.4f}\n"
            f"  Purity: {rho.purity():.4f}\n\n"
            f"Ternary Compression:\n"
            f"  Distortion: {stats['distortion']:.4f}\n"
            f"  Relative Error: {stats['relative_error']:.4f}\n"
            f"  Sparsity: {stats['sparsity']:.2%}\n"
            f"  Bits/Weight: {stats['bits_per_weight']:.2f}\n"
            f"  Compression: {1/stats['compression_ratio']:.0f}x"
        )
        return result, fig

    # --- Build Dashboard ---
    with gr.Blocks(
        title="IGQK - Quantum Compression Dashboard",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# IGQK - Information-Geometric Quantum Compression")
        gr.Markdown("Neural network compression via quantum mechanics on statistical manifolds.")

        with gr.Tabs():
            with gr.Tab("Compress"):
                with gr.Row():
                    with gr.Column(scale=1):
                        comp_file = gr.File(label="Upload Model (.pt)")
                        comp_method = gr.Radio(
                            ["Ternary (TLGT)", "Wavelet (HLWT)", "Sparse"],
                            value="Ternary (TLGT)",
                            label="Compression Method",
                        )
                        comp_ratio = gr.Slider(0.1, 0.9, value=0.5, label="Keep Ratio")
                        comp_btn = gr.Button("Compress", variant="primary")
                    with gr.Column(scale=2):
                        comp_output = gr.Textbox(label="Results", lines=15)
                        comp_plot = gr.Plot(label="Weight Distribution")
                        comp_download = gr.File(label="Download Compressed Model")

                comp_btn.click(
                    compress_weights,
                    inputs=[comp_file, comp_method, comp_ratio],
                    outputs=[comp_output, comp_plot, comp_download],
                )

            with gr.Tab("Analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ana_file = gr.File(label="Upload Model (.pt)")
                        ana_btn = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=2):
                        ana_output = gr.Textbox(label="Analysis", lines=15)
                        ana_plot = gr.Plot(label="Weight Analysis")

                ana_btn.click(
                    analyze_model,
                    inputs=[ana_file],
                    outputs=[ana_output, ana_plot],
                )

            with gr.Tab("Quick Demo"):
                with gr.Row():
                    with gr.Column(scale=1):
                        demo_method = gr.Radio(
                            ["ternary", "lowrank", "sparse"],
                            value="ternary",
                            label="Compression",
                        )
                        demo_hbar = gr.Slider(0.01, 1.0, value=0.1, label="ℏ (Quantum Uncertainty)")
                        demo_gamma = gr.Slider(0.001, 0.1, value=0.01, label="γ (Damping)")
                        demo_btn = gr.Button("Run Demo", variant="primary")
                    with gr.Column(scale=2):
                        demo_output = gr.Textbox(label="Results", lines=10)
                        demo_plot = gr.Plot(label="Compression Visualization")

                demo_btn.click(
                    run_demo,
                    inputs=[demo_method, demo_hbar, demo_gamma],
                    outputs=[demo_output, demo_plot],
                )

    return demo
