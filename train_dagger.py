import argparse
from pathlib import Path
import shutil

from train import train_policy


def train_dagger(
    iteration,
    dataset_path="aggregated_dagger_data.npz",
    save_path="dagger_policy.pth",
    epochs=50,
    batch_size=256,
    validation_fraction=0.1,
    seed=42,
    learning_rate=1e-3,
    metrics_path=None,
    label="dagger",
):
    metrics_path = metrics_path or f"dagger_training_metrics_iter_{iteration}.json"
    metrics = train_policy(
        dataset_path=dataset_path,
        save_path=save_path,
        epochs=epochs,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=seed,
        learning_rate=learning_rate,
        metrics_path=metrics_path,
        metadata={"algorithm": "dagger", "policy_label": label, "iteration": iteration},
    )
    checkpoint = Path(save_path)
    archived_checkpoint = checkpoint.with_name(
        f"{checkpoint.stem}_iter_{iteration}{checkpoint.suffix}"
    )
    shutil.copy2(checkpoint, archived_checkpoint)
    print(f"[DAgger Trainer] Iteration checkpoint: '{archived_checkpoint}'")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DAgger on the aggregated oracle labels")
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--dataset", default="aggregated_dagger_data.npz")
    parser.add_argument("--output", default="dagger_policy.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--label", default="dagger")
    args = parser.parse_args()
    train_dagger(
        args.iter, args.dataset, args.output, args.epochs, args.batch_size,
        args.validation_fraction, args.seed, args.learning_rate,
        args.metrics_output, args.label,
    )
