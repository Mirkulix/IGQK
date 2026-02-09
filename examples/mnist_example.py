"""
MNIST Example using IGQK Training

This example demonstrates the complete IGQK pipeline:
1. Train a neural network on MNIST
2. Use quantum gradient flow for optimization
3. Compress the model using ternary weights
4. Evaluate compressed model performance

Algorithm 1 from IGQK Theory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from igqk.integration.pytorch import IGQKTrainer


# Define simple neural network for MNIST
class SimpleNet(nn.Module):
    """Simple fully-connected network for MNIST."""

    def __init__(self, hidden_dim=128):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_mnist_data(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


def plot_training_history(history):
    """Plot training metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    if history.get('val_loss'):
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Entropy
    axes[1].plot(history['entropy'], label='Entropy', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Von Neumann Entropy')
    axes[1].set_title('Quantum State Entropy')
    axes[1].legend()
    axes[1].grid(True)

    # Purity
    axes[2].plot(history['purity'], label='Purity', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Purity')
    axes[2].set_title('Quantum State Purity')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('igqk_training_history.png', dpi=150)
    print("Saved training history plot to 'igqk_training_history.png'")


def main():
    """Main training function."""
    print("=" * 80)
    print("IGQK Training Example - MNIST")
    print("Information-Geometric Quantum Compression")
    print("=" * 80)

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=128)
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = SimpleNet(hidden_dim=128)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Initialize IGQK trainer
    print("\nInitializing IGQK trainer...")
    print("  Quantum parameters:")
    print("    ℏ (hbar) = 0.1 (quantum uncertainty)")
    print("    γ (gamma) = 0.01 (damping)")
    print("    Compression: ternary weights {-1, 0, +1}")

    trainer = IGQKTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        hbar=0.1,
        gamma=0.01,
        compression_type='ternary',
        fisher_diagonal=True,
        device=device
    )

    # Train model
    print("\n" + "=" * 80)
    print("Training with Quantum Gradient Flow")
    print("=" * 80)

    num_epochs = 5  # Small number for demo
    trainer.train(
        num_epochs=num_epochs,
        compute_fisher_every=1  # Compute Fisher matrix every epoch (diagonal approximation)
    )

    # Evaluate final model
    print("\n" + "=" * 80)
    print("Final Model Evaluation")
    print("=" * 80)

    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFull-precision model:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")

    # Compress model
    print("\n" + "=" * 80)
    print("Compressing Model with Ternary Weights")
    print("=" * 80)

    compressed_model = trainer.compress()

    # Evaluate compressed model
    print("\nCompressed model results:")
    comp_loss, comp_acc = trainer.evaluate(test_loader)
    print(f"  Test Loss: {comp_loss:.4f}")
    print(f"  Test Accuracy: {comp_acc * 100:.2f}%")

    # Compression statistics
    compression_ratio = 2.0 / 32.0  # Ternary encoding
    print(f"\n  Compression ratio: {compression_ratio:.4f} (~1/16)")
    print(f"  Accuracy drop: {(test_acc - comp_acc) * 100:.2f}%")

    # Plot training history
    print("\nGenerating plots...")
    plot_training_history(trainer.history)

    # Save models
    print("\nSaving models...")
    torch.save(model.state_dict(), 'igqk_model_full.pth')
    torch.save(compressed_model.state_dict(), 'igqk_model_compressed.pth')
    print("  Saved 'igqk_model_full.pth'")
    print("  Saved 'igqk_model_compressed.pth'")

    # Summary
    print("\n" + "=" * 80)
    print("IGQK Training Complete!")
    print("=" * 80)
    print("\nKey Results:")
    print(f"  ✓ Trained on {len(train_loader.dataset)} samples")
    print(f"  ✓ Test accuracy: {test_acc * 100:.2f}%")
    print(f"  ✓ Compressed to {compression_ratio:.4f} of original size")
    print(f"  ✓ Compressed accuracy: {comp_acc * 100:.2f}%")
    print(f"\nTheoretical Foundation:")
    print(f"  - Statistical manifold with Fisher metric")
    print(f"  - Quantum gradient flow: dρ/dt = -i[H,ρ] - γ{{G⁻¹∇L,ρ}}")
    print(f"  - Projection onto ternary submanifold N = {{-1, 0, +1}}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
