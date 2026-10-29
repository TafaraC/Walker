import argparse

import numpy as np

from bc_data import load_dataset, save_dataset_atomic, save_json_atomic


def split_expert_data(
    source, train_output, test_output, metadata_output,
    test_fraction=0.1, block_size=250, seed=42,
):
    observations, actions = load_dataset(source)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # Contiguous blocks preserve local trajectory structure and reduce leakage
    # compared with independently shuffling highly correlated control cycles.
    blocks = [np.arange(start, min(start + block_size, len(observations)))
              for start in range(0, len(observations), block_size)]
    if len(blocks) < 2:
        raise ValueError("Dataset must contain at least two trajectory blocks")
    rng = np.random.default_rng(seed)
    test_block_count = min(len(blocks) - 1, max(1, round(len(blocks) * test_fraction)))
    test_blocks = set(rng.choice(len(blocks), size=test_block_count, replace=False).tolist())
    test_indices = np.concatenate([block for index, block in enumerate(blocks) if index in test_blocks])
    train_indices = np.concatenate([block for index, block in enumerate(blocks) if index not in test_blocks])

    save_dataset_atomic(train_output, observations[train_indices], actions[train_indices])
    save_dataset_atomic(test_output, observations[test_indices], actions[test_indices])
    save_json_atomic(metadata_output, {
        "source": source,
        "train_output": train_output,
        "test_output": test_output,
        "seed": seed,
        "test_fraction": test_fraction,
        "block_size": block_size,
        "total_samples": len(observations),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "total_blocks": len(blocks),
        "test_blocks": sorted(test_blocks),
    })
    print(f"[Expert Split] Train: {len(train_indices)} | Test: {len(test_indices)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a fixed held-out oracle split")
    parser.add_argument("--source", default="expert_walk_data.npz")
    parser.add_argument("--train-output", default="expert_train_data.npz")
    parser.add_argument("--test-output", default="expert_test_data.npz")
    parser.add_argument("--metadata-output", default="expert_split_metadata.json")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_expert_data(
        args.source, args.train_output, args.test_output, args.metadata_output,
        args.test_fraction, args.block_size, args.seed,
    )
