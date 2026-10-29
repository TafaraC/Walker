import os
import argparse
import json
import random
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from bc_data import load_dataset

class BCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_policy(
    dataset_path="expert_walk_data.npz",
    save_path="bc_walk_policy.pth",
    epochs=100,
    batch_size=256,
    validation_fraction=0.1,
    seed=42,
    learning_rate=1e-3,
    metrics_path="bc_training_metrics.json",
    metadata=None,
):
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset file '{dataset_path}' not found!")
        raise FileNotFoundError(dataset_path)

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=================================================")
    print(f" STARTING OFFLINE TRAINING ON {device.type.upper()}")
    print(f"=================================================")

    X, Y = load_dataset(dataset_path)
    if len(X) < 2:
        raise ValueError("At least two samples are required for train/validation splitting")

    print(f"Dataset Loaded: {X.shape[0]} state-action samples.")
    print(f"Observation Dim: {X.shape[1]} | Action Dim: {Y.shape[1]}")

    generator = torch.Generator().manual_seed(seed)
    permutation = np.random.default_rng(seed).permutation(len(X))
    validation_size = max(1, int(round(len(X) * validation_fraction)))
    validation_size = min(validation_size, len(X) - 1)
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]

    train_dataset = TensorDataset(
        torch.from_numpy(X[training_indices]), torch.from_numpy(Y[training_indices])
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(X[validation_indices]), torch.from_numpy(Y[validation_indices])
    )
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    model = BCPolicy(X.shape[1], Y.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    best_state = None

    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(validation_dataset)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        avg_loss = total_loss / len(train_dataset)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                loss = criterion(model(batch_x), batch_y)
                validation_loss += loss.item() * len(batch_x)
        validation_loss /= len(validation_dataset)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.6f} "
                f"| Validation Loss: {validation_loss:.6f}"
            )

    torch.save(best_state, save_path)
    model.load_state_dict(best_state)
    model.to(device).eval()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    target_squared_error_sum = 0.0
    per_action_squared_error = torch.zeros(Y.shape[1], device=device)
    per_action_absolute_error = torch.zeros(Y.shape[1], device=device)
    target_mean = torch.from_numpy(Y[validation_indices].mean(axis=0)).to(device)
    with torch.no_grad():
        for batch_x, batch_y in validation_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            error = model(batch_x) - batch_y
            squared_error_sum += error.square().sum().item()
            absolute_error_sum += error.abs().sum().item()
            target_squared_error_sum += (batch_y - target_mean).square().sum().item()
            per_action_squared_error += error.square().sum(dim=0)
            per_action_absolute_error += error.abs().sum(dim=0)

    element_count = len(validation_dataset) * Y.shape[1]
    validation_mse = squared_error_sum / element_count
    metrics = {
        "algorithm": "behavioral_cloning",
        "oracle": "FC Portugal Walk policy",
        "dataset": str(Path(dataset_path).resolve()),
        "checkpoint": str(Path(save_path).resolve()),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "sample_count": len(X),
        "training_sample_count": len(train_dataset),
        "validation_sample_count": len(validation_dataset),
        "observation_dim": X.shape[1],
        "action_dim": Y.shape[1],
        "validation_mse": validation_mse,
        "validation_rmse": validation_mse ** 0.5,
        "validation_mae": absolute_error_sum / element_count,
        "validation_r2": (
            1.0 - squared_error_sum / target_squared_error_sum
            if target_squared_error_sum > 0
            else None
        ),
        "per_action_rmse": (
            per_action_squared_error.div(len(validation_dataset)).sqrt().cpu().tolist()
        ),
        "per_action_mae": (
            per_action_absolute_error.div(len(validation_dataset)).cpu().tolist()
        ),
    }
    if metadata:
        metrics.update(metadata)
    metrics_target = Path(metrics_path)
    metrics_target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", prefix=f".{metrics_target.name}.", suffix=".tmp",
        dir=metrics_target.parent, delete=False
    ) as temporary_file:
        json.dump(metrics, temporary_file, indent=2)
        temporary_file.write("\n")
        temporary_name = temporary_file.name
    os.replace(temporary_name, metrics_target)
    print(
        f"\n[Trainer] Training complete. Best validation loss: "
        f"{best_validation_loss:.6f}. Model saved to '{save_path}'"
    )
    print(f"[Trainer] Offline imitation metrics saved to '{metrics_path}'")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a behavioral-cloning walk policy")
    parser.add_argument("--dataset", default="expert_walk_data.npz")
    parser.add_argument("--output", default="bc_walk_policy.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", default="bc_training_metrics.json")
    args = parser.parse_args()
    train_policy(
        dataset_path=args.dataset,
        save_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        learning_rate=args.learning_rate,
        metrics_path=args.metrics_output,
    )
