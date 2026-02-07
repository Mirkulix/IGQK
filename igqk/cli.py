"""
IGQK Command-Line Interface.

Usage:
    igqk train --model resnet18 --dataset mnist --compression ternary
    igqk compress --checkpoint model.pt --method ternary --output compressed.pt
    igqk evaluate --checkpoint compressed.pt --dataset mnist
    igqk serve --host 0.0.0.0 --port 8000
    igqk dashboard --port 7860
    igqk info
"""

import argparse
import json
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        prog="igqk",
        description="IGQK - Information-Geometric Quantum Compression",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train a model with IGQK")
    train_parser.add_argument("--model", default="simple_fc", help="Model architecture")
    train_parser.add_argument("--dataset", default="mnist", help="Dataset name")
    train_parser.add_argument(
        "--compression", default="ternary", choices=["ternary", "lowrank", "sparse"]
    )
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--hbar", type=float, default=0.1, help="Quantum uncertainty")
    train_parser.add_argument("--gamma", type=float, default=0.01, help="Damping parameter")
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (dt)")
    train_parser.add_argument("--output", default="igqk_model.pt", help="Output checkpoint")
    train_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # --- compress ---
    compress_parser = subparsers.add_parser("compress", help="Compress a trained model")
    compress_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    compress_parser.add_argument(
        "--method", default="ternary", choices=["ternary", "lowrank", "sparse", "wavelet"]
    )
    compress_parser.add_argument("--rank", type=int, default=10, help="Rank for lowrank")
    compress_parser.add_argument("--sparsity", type=float, default=0.1, help="Sparsity ratio")
    compress_parser.add_argument("--output", default="compressed_model.pt")

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--dataset", default="mnist")
    eval_parser.add_argument("--batch-size", type=int, default=256)

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start REST API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--workers", type=int, default=1)
    serve_parser.add_argument("--reload", action="store_true")

    # --- dashboard ---
    dash_parser = subparsers.add_parser("dashboard", help="Start Gradio dashboard")
    dash_parser.add_argument("--host", default="0.0.0.0")
    dash_parser.add_argument("--port", type=int, default=7860)
    dash_parser.add_argument("--share", action="store_true")

    # --- info ---
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "compress":
        _cmd_compress(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)
    elif args.command == "info":
        _cmd_info()


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _cmd_train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from igqk.integration.pytorch import IGQKTrainer

    device = _resolve_device(args.device)
    print(f"[IGQK] Training with quantum compression")
    print(f"  Model: {args.model} | Dataset: {args.dataset}")
    print(f"  Compression: {args.compression} | Device: {device}")
    print(f"  hbar={args.hbar}, gamma={args.gamma}, epochs={args.epochs}")

    # Dataset
    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
        val_ds = datasets.MNIST("./data", train=False, transform=transform)
        in_features, num_classes = 784, 10
    elif args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        val_ds = datasets.CIFAR10("./data", train=False, transform=transform)
        in_features, num_classes = 3072, 10
    else:
        print(f"[Error] Unknown dataset: {args.dataset}")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    if args.model == "simple_fc":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    else:
        print(f"[Error] Unknown model: {args.model}")
        sys.exit(1)

    # Train
    trainer = IGQKTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hbar=args.hbar,
        gamma=args.gamma,
        compression_type=args.compression,
        device=device,
    )
    trainer.train(num_epochs=args.epochs)

    # Compress and save
    compressed_model = trainer.compress()
    torch.save({
        "model_state_dict": compressed_model.state_dict(),
        "compression_type": args.compression,
        "hbar": args.hbar,
        "gamma": args.gamma,
        "history": trainer.history,
    }, args.output)
    print(f"[IGQK] Model saved to {args.output}")


def _cmd_compress(args):
    import torch

    from igqk.compression.projection import OptimalProjection
    from igqk.theory.hlwt import HybridLaplaceWavelet
    from igqk.theory.tlgt import TernaryLieGroup

    print(f"[IGQK] Compressing {args.checkpoint} with method={args.method}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    compressed_state = {}
    total_original = 0
    total_compressed = 0

    for name, param in state_dict.items():
        original_size = param.numel() * 4  # float32 = 4 bytes
        total_original += original_size

        if args.method == "ternary":
            tlgt = TernaryLieGroup(param.numel())
            compressed, scale = tlgt.quantize(param)
            compressed_state[name] = compressed
            # Ternary: ~2 bits per weight
            total_compressed += param.numel() * 2 // 8
        elif args.method == "wavelet":
            hlwt = HybridLaplaceWavelet()
            compressed, ratio = hlwt.compress(param, keep_ratio=0.3)
            compressed_state[name] = compressed
            total_compressed += int(original_size * ratio)
        elif args.method == "lowrank":
            proj = OptimalProjection(submanifold_type="lowrank", rank=args.rank)
            compressed_state[name] = proj.projector.project(param.flatten()).view_as(param)
            total_compressed += original_size  # stored as float but lower effective rank
        elif args.method == "sparse":
            proj = OptimalProjection(submanifold_type="sparse", sparsity=args.sparsity)
            compressed_state[name] = proj.projector.project(param.flatten()).view_as(param)
            total_compressed += int(original_size * args.sparsity)

    ratio = total_compressed / total_original if total_original > 0 else 1.0
    print(f"  Original size:    {total_original / 1024:.1f} KB")
    print(f"  Compressed size:  {total_compressed / 1024:.1f} KB")
    print(f"  Compression ratio: {ratio:.4f} ({1/ratio:.1f}x)")

    torch.save({"model_state_dict": compressed_state, "method": args.method}, args.output)
    print(f"[IGQK] Compressed model saved to {args.output}")


def _cmd_evaluate(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    print(f"[IGQK] Evaluating {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Infer model architecture from state dict
    keys = list(state_dict.keys())
    first_weight = state_dict[keys[0]]
    in_features = first_weight.shape[1] if first_weight.dim() == 2 else first_weight.shape[0]
    last_weight = state_dict[keys[-2]] if len(keys) > 1 else first_weight
    num_classes = last_weight.shape[0] if last_weight.dim() == 2 else 10

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_ds = datasets.MNIST("./data", train=False, transform=transform)
    else:
        print(f"[Error] Unknown dataset: {args.dataset}")
        sys.exit(1)

    loader = DataLoader(test_ds, batch_size=args.batch_size)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Weight statistics
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    ternary = sum(
        ((p == -1) | (p == 0) | (p == 1)).all().item()
        for p in model.parameters()
    )
    print(f"  Parameters: {total_params:,}")
    print(f"  Zero weights: {zero_params:,} ({100*zero_params/total_params:.1f}%)")


def _cmd_serve(args):
    try:
        import uvicorn
    except ImportError:
        print("[Error] Install API dependencies: pip install igqk[api]")
        sys.exit(1)

    print(f"[IGQK] Starting API server on {args.host}:{args.port}")
    uvicorn.run(
        "igqk.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


def _cmd_dashboard(args):
    try:
        from igqk.dashboard.app import create_dashboard
    except ImportError:
        print("[Error] Install dashboard dependencies: pip install igqk[dashboard]")
        sys.exit(1)

    print(f"[IGQK] Starting dashboard on {args.host}:{args.port}")
    demo = create_dashboard()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


def _cmd_info():
    import igqk

    print(f"IGQK v{igqk.__version__}")
    print(f"  Information-Geometric Quantum Compression")
    print()

    try:
        import torch
        print(f"  PyTorch:      {torch.__version__}")
        print(f"  CUDA:         {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  PyTorch:      NOT INSTALLED")

    try:
        import numpy
        print(f"  NumPy:        {numpy.__version__}")
    except ImportError:
        print("  NumPy:        NOT INSTALLED")

    try:
        import scipy
        print(f"  SciPy:        {scipy.__version__}")
    except ImportError:
        print("  SciPy:        NOT INSTALLED")

    try:
        import fastapi
        print(f"  FastAPI:      {fastapi.__version__}")
    except ImportError:
        print("  FastAPI:      not installed (pip install igqk[api])")

    try:
        import gradio
        print(f"  Gradio:       {gradio.__version__}")
    except ImportError:
        print("  Gradio:       not installed (pip install igqk[dashboard])")

    print()
    print("  Commands: train, compress, evaluate, serve, dashboard, info")


if __name__ == "__main__":
    main()
