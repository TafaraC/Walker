"""Dataset utilities shared by the behavioral-cloning collectors and trainer."""

from pathlib import Path
import json
import os
import tempfile

import numpy as np


REQUIRED_KEYS = ("obs", "actions")
OWN_HALF_X_BOUNDS = (-6.5, -0.5)
FIELD_Y_BOUNDS = (-6.5, 6.5)


def sample_own_half_target():
    """Sample a target inside the buffered playable area of the player's half."""
    return np.array(
        [
            np.random.uniform(*OWN_HALF_X_BOUNDS),
            np.random.uniform(*FIELD_Y_BOUNDS),
        ],
        dtype=np.float32,
    )


def sample_short_own_half_target(current_pos, min_distance=1.0, max_distance=2.0):
    """Sample a nearby target without crossing midfield or field boundaries."""
    current_pos = np.asarray(current_pos, dtype=np.float32)
    for _ in range(100):
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        distance = np.random.uniform(min_distance, max_distance)
        target = current_pos + distance * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        if (
            OWN_HALF_X_BOUNDS[0] <= target[0] <= OWN_HALF_X_BOUNDS[1]
            and FIELD_Y_BOUNDS[0] <= target[1] <= FIELD_Y_BOUNDS[1]
        ):
            return target

    # This is only reachable when the reported player position is outside the
    # playable sampling area (for example, immediately after a stale reset).
    return np.array(
        [
            np.clip(current_pos[0], *OWN_HALF_X_BOUNDS),
            np.clip(current_pos[1], *FIELD_Y_BOUNDS),
        ],
        dtype=np.float32,
    )


def load_dataset(path):
    """Load and validate an imitation-learning state/action dataset."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        obs = np.asarray(archive["obs"], dtype=np.float32)
        actions = np.asarray(archive["actions"], dtype=np.float32)

    if obs.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"Expected 2-D obs/actions arrays, got {obs.shape} and {actions.shape}"
        )
    if len(obs) != len(actions):
        raise ValueError(
            f"Observation/action counts differ: {len(obs)} != {len(actions)}"
        )
    if len(obs) == 0:
        raise ValueError(f"{path} contains no samples")
    if not np.isfinite(obs).all() or not np.isfinite(actions).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return obs, actions


def save_dataset_atomic(path, obs, actions):
    """Write a dataset atomically so an interrupted save cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary_name, obs=obs, actions=actions)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)


def save_json_atomic(path, payload):
    """Atomically save a JSON metrics or metadata document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)


def append_dataset(main_path, chunk_path):
    """Validate a collector chunk and atomically append it to the main dataset."""
    new_obs, new_actions = load_dataset(chunk_path)
    main_path = Path(main_path)
    if main_path.exists():
        old_obs, old_actions = load_dataset(main_path)
        if old_obs.shape[1:] != new_obs.shape[1:]:
            raise ValueError(
                f"Observation dimensions differ: {old_obs.shape[1:]} != {new_obs.shape[1:]}"
            )
        if old_actions.shape[1:] != new_actions.shape[1:]:
            raise ValueError(
                f"Action dimensions differ: {old_actions.shape[1:]} != {new_actions.shape[1:]}"
            )
        new_obs = np.concatenate((old_obs, new_obs), axis=0)
        new_actions = np.concatenate((old_actions, new_actions), axis=0)

    save_dataset_atomic(main_path, new_obs, new_actions)
    Path(chunk_path).unlink()
    return len(new_obs)
