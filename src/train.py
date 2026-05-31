import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Import the model architecture
from model import resnet10_1d, resnet18_1d

class PPGDataset(Dataset):
    """Custom Dataset for loading preprocessed PPG signal windows and BPM labels."""
    def __init__(self, x_path, y_path):
        print(f"Loading data from {x_path} and {y_path}...")
        self.x = np.load(x_path)
        self.y = np.load(y_path)
        
        # Convert to PyTorch tensors
        self.x = torch.from_numpy(self.x).float()  # shape (N, 1, 1000)
        self.y = torch.from_numpy(self.y).float().unsqueeze(1)  # shape (N, 1)
        
        print(f"Loaded {len(self.x)} samples. X shape: {self.x.shape}, Y shape: {self.y.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the given dataloader and returns MAE and RMSE."""
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Compute MAE
            mae = torch.abs(outputs - targets).sum().item()
            total_mae += mae
            
            # Compute MSE
            mse = torch.pow(outputs - targets, 2).sum().item()
            total_mse += mse
            
            total_samples += targets.size(0)
            
    val_mae = total_mae / total_samples
    val_rmse = np.sqrt(total_mse / total_samples)
    return val_mae, val_rmse


def train(args):
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Ensure output directories exist
    os.makedirs(args.model_dir, exist_ok=True)

    # Load datasets
    train_dataset = PPGDataset(
        os.path.join(args.data_dir, "train_x.npy"),
        os.path.join(args.data_dir, "train_y.npy")
    )
    val_dataset = PPGDataset(
        os.path.join(args.data_dir, "val_x.npy"),
        os.path.join(args.data_dir, "val_y.npy")
    )
    test_dataset = PPGDataset(
        os.path.join(args.data_dir, "test_x.npy"),
        os.path.join(args.data_dir, "test_y.npy")
    )

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Select Model Architecture
    if args.arch == "resnet10":
        model = resnet10_1d(num_classes=1)
    elif args.arch == "resnet18":
        model = resnet18_1d(num_classes=1)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
        
    model = model.to(device)
    print(f"Model architecture: {args.arch}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Loss function and Optimizer
    # We optimize for MAE (L1Loss) directly since the metric is MAE
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Logging history
    train_losses = []
    val_losses = []
    val_rmses = []
    
    best_val_mae = float("inf")
    checkpoint_path = os.path.join(args.model_dir, "best_model.pth")

    print("\n=== Start Training ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            
        epoch_train_loss = running_loss / total_samples
        
        # Evaluate on validation set
        epoch_val_mae, epoch_val_rmse = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(epoch_val_mae)
        
        # Log metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_mae)
        val_rmses.append(epoch_val_rmse)
        
        print(f"Epoch [{epoch}/{args.epochs}] - "
              f"Train MAE: {epoch_train_loss:.4f} | "
              f"Val MAE: {epoch_val_mae:.4f} | "
              f"Val RMSE: {epoch_val_rmse:.4f}")
        
        # Save best model
        if epoch_val_mae < best_val_mae:
            best_val_mae = epoch_val_mae
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best model checkpoint to {checkpoint_path} (Val MAE: {best_val_mae:.4f})")

    print("\n=== Training Complete ===")
    print(f"Best Validation MAE: {best_val_mae:.4f}")

    # Load best checkpoint for testing
    print(f"Loading best checkpoint from {checkpoint_path} for final evaluation...")
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Evaluate on test set
    test_mae, test_rmse = evaluate(model, test_loader, criterion, device)
    print(f"\n=== Final Test Evaluation ===")
    print(f"Test MAE:  {test_mae:.4f} BPM")
    print(f"Test RMSE: {test_rmse:.4f} BPM")

    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss (MAE)", marker="o")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss (MAE)", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE) - BPM")
    plt.title(f"1D ResNet PPG Training Loss Curves ({args.arch})")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(args.model_dir, "loss_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved learning curves plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D ResNet for PPG Heart Rate Estimation")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed data files")
    parser.add_argument("--model_dir", type=str, default="models", help="Path to save models and curves")
    parser.add_argument("--arch", type=str, default="resnet10", choices=["resnet10", "resnet18"], help="ResNet architecture variant")
    parser.add_argument("--epochs", type=str, default="15", help="Number of training epochs")  # string to allow flexible parsing or custom inputs if needed, we parse to int
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for L2 regularization")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Cast epoch back to integer in case
    args.epochs = int(args.epochs)
    
    train(args)
