import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
import torch

from bc_data import load_dataset, save_json_atomic
from gail_models import GAILActorCritic
from train import BCPolicy


DT = 0.02


def confidence_interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half_width = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "ci95_low": mean - half_width, "ci95_high": mean + half_width, "n": len(values)}


def smooth(sequence, window=5):
    if len(sequence) < window or window <= 1:
        return sequence
    padding = window // 2
    padded = np.pad(sequence, ((padding, padding), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack([np.convolve(padded[:, joint], kernel, mode="valid") for joint in range(sequence.shape[1])], axis=1)


def jerk_cost(joints_deg, episode_ids, smoothing_window=5):
    integrated = 0.0
    valid_steps = 0
    for episode in np.unique(episode_ids):
        angles = np.deg2rad(joints_deg[episode_ids == episode].astype(np.float64))
        if len(angles) < 4:
            continue
        angles = smooth(angles, smoothing_window)
        jerk = np.diff(angles, n=3, axis=0) / (DT ** 3)
        integrated += float(np.square(jerk).sum() * DT)
        valid_steps += len(jerk)
    duration = valid_steps * DT
    return integrated, integrated / duration if duration else 0.0


def sliced_wasserstein(learned, expert, projections=256, seed=12345):
    expert_mean = expert.mean(axis=0)
    expert_std = expert.std(axis=0)
    expert_std[expert_std < 1e-6] = 1.0
    learned = (learned - expert_mean) / expert_std
    expert = (expert - expert_mean) / expert_std
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, expert.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    learned_projection = np.sort(learned @ directions.T, axis=0)
    expert_projection = np.sort(expert @ directions.T, axis=0)
    quantiles = np.linspace(0.0, 1.0, max(len(learned), len(expert)))
    source_q = np.linspace(0.0, 1.0, len(learned))
    expert_q = np.linspace(0.0, 1.0, len(expert))
    distances = []
    for index in range(projections):
        left = np.interp(quantiles, source_q, learned_projection[:, index])
        right = np.interp(quantiles, expert_q, expert_projection[:, index])
        distances.append(np.mean(np.abs(left - right)))
    return float(np.mean(distances))


def offline_action_mse(kind, model_path, observations, oracle_actions, random_low, random_high, seed=123):
    if kind == "oracle":
        predictions = oracle_actions
    elif kind == "random":
        predictions = np.random.default_rng(seed).uniform(
            random_low, random_high, size=oracle_actions.shape
        ).astype(np.float32)
    else:
        if kind in ("bc", "dagger"):
            model = BCPolicy(observations.shape[1], oracle_actions.shape[1])
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            predict = model
        else:
            model = GAILActorCritic(observations.shape[1], oracle_actions.shape[1])
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            predict = model.actor_mean
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(observations), 1024):
                parts.append(predict(torch.from_numpy(observations[start:start + 1024])).numpy())
        predictions = np.concatenate(parts)
    errors = predictions - oracle_actions
    return float(np.square(errors).mean()), np.square(errors).mean(axis=0).tolist()


def iteration_for(label, prefix):
    match = re.fullmatch(rf"{prefix}_iter_(\d+)", label)
    return int(match.group(1)) if match else None


def resource_counts(label, kind):
    if kind == "bc":
        path = Path("artifacts/bc/training_metrics.json")
        if not path.exists():
            path = Path("bc_training_metrics.json")
        if path.exists():
            return {"oracle_queries": json.loads(path.read_text())["sample_count"], "environment_steps": 0}
    if kind == "dagger":
        base_label = re.sub(r"_iter_\d+$", "", label)
        iteration = iteration_for(label, base_label)
        candidates = sorted(Path("artifacts", base_label).glob("training_metrics_iter_*.json"))
        if not candidates:
            candidates = sorted(Path(".").glob(f"{base_label}_training_metrics_iter_*.json"))
        if iteration is None and candidates:
            iteration = max(int(re.search(r"(\d+)$", path.stem).group(1)) for path in candidates)
        path = Path("artifacts", base_label, f"training_metrics_iter_{iteration}.json") if iteration is not None else None
        if path is not None and not path.exists():
            path = Path(f"{base_label}_training_metrics_iter_{iteration}.json")
        if path and path.exists():
            training = json.loads(path.read_text())
            rollout_steps = 0
            metric_paths = list(Path("artifacts", base_label).glob("rollout_metrics_iter_*.json"))
            if not metric_paths:
                metric_paths = list(Path(".").glob(f"{base_label}_rollout_metrics_iter_*.json"))
            for metric_path in metric_paths:
                metric = json.loads(metric_path.read_text())
                if metric["iteration"] <= iteration:
                    rollout_steps += metric["rollout_samples"]
            return {"oracle_queries": training["sample_count"], "environment_steps": rollout_steps}
    if kind == "gail":
        base_label = re.sub(r"_iter_\d+$", "", label)
        iteration = iteration_for(label, base_label)
        metrics = []
        metric_paths = list(Path("artifacts", base_label).glob("rollout_metrics_iter_*.json"))
        if not metric_paths:
            metric_paths = list(Path(".").glob(f"{base_label}_rollout_metrics_iter_*.json"))
        if not metric_paths:
            metric_paths = list(Path("experiments").glob(
                f"**/{base_label}_rollout_metrics_iter_*.json"
            ))
        for path in metric_paths:
            item = json.loads(path.read_text())
            if iteration is None or item["iteration"] <= iteration:
                metrics.append(item)
        return {"oracle_queries": 0, "environment_steps": sum(item["rollout_samples"] for item in metrics)}
    return {"oracle_queries": 0, "environment_steps": 0}


def summarize(raw_dir, expert_test_path, expert_train_path, output_json, output_csv, projections, smoothing_window):
    raw_dir = Path(raw_dir)
    summaries = [json.loads(path.read_text()) for path in raw_dir.glob("*_seed_*.json")]
    if not summaries:
        raise ValueError(f"No evaluation summaries found in {raw_dir}")
    by_label = {}
    for summary in summaries:
        by_label.setdefault(summary["policy_label"], []).append(summary)
    if "random" not in by_label or "oracle" not in by_label:
        raise ValueError("Random and oracle baseline evaluations are required")
    environment_ids = {item.get("environment_id", "nominal") for item in summaries}
    field_slips = {float(item.get("field_contact_slip", 0.1)) for item in summaries}
    if len(environment_ids) != 1 or len(field_slips) != 1:
        raise ValueError(
            f"Evaluation directory mixes environments: ids={sorted(environment_ids)}, "
            f"field_slips={sorted(field_slips)}"
        )

    baseline = {}
    for label in ("random", "oracle"):
        baseline[label] = {item["seed"]: item["return_target_progress_m"] for item in by_label[label]}
    required_seeds = sorted({item["seed"] for items in by_label.values() for item in items})
    missing_baselines = {
        label: [seed for seed in required_seeds if seed not in baseline[label]]
        for label in ("random", "oracle")
    }
    missing_baselines = {label: seeds for label, seeds in missing_baselines.items() if seeds}
    if missing_baselines:
        details = "; ".join(
            f"{label}: {', '.join(map(str, seeds))}"
            for label, seeds in missing_baselines.items()
        )
        raise ValueError(
            "Missing per-seed normalized-return baselines (" + details + "). "
            "Resume with RESET_EVALUATION=0 RESUME_EVALUATION=1 ./run_evaluation.sh"
        )
    test_obs, test_actions = load_dataset(expert_test_path)
    _, train_actions = load_dataset(expert_train_path)
    random_low, random_high = train_actions.min(axis=0), train_actions.max(axis=0)
    oracle_traces = {}
    for item in by_label["oracle"]:
        with np.load(item["trace"], allow_pickle=False) as trace:
            oracle_traces[item["seed"]] = np.asarray(trace["joint_positions_deg"], dtype=np.float64)

    per_run = []
    for label, items in sorted(by_label.items()):
        for item in sorted(items, key=lambda value: value["seed"]):
            seed = item["seed"]
            denominator = baseline["oracle"][seed] - baseline["random"][seed]
            normalized = (
                (item["return_target_progress_m"] - baseline["random"][seed]) / denominator
                if abs(denominator) > 1e-12 else None
            )
            with np.load(item["trace"], allow_pickle=False) as trace:
                joints = np.asarray(trace["joint_positions_deg"], dtype=np.float64)
                episodes = np.asarray(trace["episode_id"], dtype=np.int64)
                roll = np.asarray(trace["torso_roll_deg"], dtype=np.float64)
                pitch = np.asarray(trace["torso_pitch_deg"], dtype=np.float64)
                head = np.asarray(trace["head_height_m"], dtype=np.float64)
            jerk_integrated, jerk_rate = jerk_cost(joints, episodes, smoothing_window)
            row = dict(item)
            row.update({
                "normalized_return": normalized,
                "trajectory_sliced_wasserstein": sliced_wasserstein(
                    joints, oracle_traces[seed], projections=projections
                ),
                "jerk_integrated_rad2_per_s5": jerk_integrated,
                "jerk_cost_rate_rad2_per_s6": jerk_rate,
                "torso_roll_rms_deg": float(np.sqrt(np.mean(np.square(roll)))),
                "torso_roll_peak_deg": float(np.max(np.abs(roll))),
                "torso_pitch_rms_deg": float(np.sqrt(np.mean(np.square(pitch)))),
                "torso_pitch_peak_deg": float(np.max(np.abs(pitch))),
                "mean_head_height_m": float(np.mean(head)),
                "minimum_head_height_m": float(np.min(head)),
            })
            per_run.append(row)

    aggregates = {}
    numeric_metrics = (
        "normalized_return", "return_target_progress_m", "mean_target_directed_velocity_mps",
        "mean_planar_speed_mps", "mean_forward_velocity_mps", "mean_target_direction_error", "falls_per_active_minute",
        "on_policy_oracle_action_mse", "trajectory_sliced_wasserstein",
        "jerk_integrated_rad2_per_s5", "jerk_cost_rate_rad2_per_s6",
        "torso_roll_rms_deg", "torso_pitch_rms_deg", "minimum_head_height_m",
    )
    for label, items in sorted(by_label.items()):
        rows = [row for row in per_run if row["policy_label"] == label]
        kind, model_path = items[0]["policy_kind"], items[0]["model"]
        action_mse, per_action_mse = offline_action_mse(
            kind, model_path, test_obs, test_actions, random_low, random_high
        )
        metric_summaries = {
            metric: confidence_interval([row[metric] for row in rows if row[metric] is not None])
            for metric in numeric_metrics
        }
        # Normal-approximation intervals can cross zero. Enforce the valid
        # support for metrics that are physically/statistically non-negative.
        nonnegative_metrics = (
            "mean_planar_speed_mps", "falls_per_active_minute",
            "on_policy_oracle_action_mse", "trajectory_sliced_wasserstein",
            "jerk_integrated_rad2_per_s5", "jerk_cost_rate_rad2_per_s6",
            "torso_roll_rms_deg", "torso_pitch_rms_deg", "minimum_head_height_m",
        )
        for metric_name in nonnegative_metrics:
            metric_summaries[metric_name]["ci95_low"] = max(
                0.0, metric_summaries[metric_name]["ci95_low"]
            )
        aggregates[label] = {
            "policy_kind": kind,
            "model": model_path,
            "resources": resource_counts(label, kind),
            "expert_test_action_mse": action_mse,
            "expert_test_per_action_mse": per_action_mse,
            "metrics": metric_summaries,
        }

    report = {
        "protocol": {
            "dt_seconds": DT,
            "environment_id": next(iter(environment_ids)),
            "field_contact_slip": next(iter(field_slips)),
            "expert_test_data": expert_test_path,
            "expert_train_data": expert_train_path,
            "wasserstein": f"sliced Wasserstein-1 with {projections} projections after expert standardization",
            "jerk": f"third finite difference after {smoothing_window}-sample moving average; reset boundaries excluded",
            "normalized_return_baselines": ["random", "oracle"],
        },
        "aggregates": aggregates,
        "per_run": per_run,
    }
    save_json_atomic(output_json, report)

    flat_rows = []
    for label, result in aggregates.items():
        row = {
            "policy_label": label,
            "policy_kind": result["policy_kind"],
            "oracle_queries": result["resources"]["oracle_queries"],
            "environment_steps": result["resources"]["environment_steps"],
            "expert_test_action_mse": result["expert_test_action_mse"],
        }
        for metric, stats in result["metrics"].items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_ci95_low"] = stats["ci95_low"]
            row[f"{metric}_ci95_high"] = stats["ci95_high"]
        flat_rows.append(row)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"[Evaluation] Report: {output_json}")
    print(f"[Evaluation] Table:  {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate the imitation evaluation protocol")
    parser.add_argument("--raw-dir", default="evaluation/raw")
    parser.add_argument("--expert-test", default="expert_test_data.npz")
    parser.add_argument("--expert-train", default="expert_train_data.npz")
    parser.add_argument("--output-json", default="evaluation/comparison.json")
    parser.add_argument("--output-csv", default="evaluation/comparison.csv")
    parser.add_argument("--projections", type=int, default=256)
    parser.add_argument("--smoothing-window", type=int, default=5)
    args = parser.parse_args()
    summarize(
        args.raw_dir, args.expert_test, args.expert_train,
        args.output_json, args.output_csv, args.projections,
        args.smoothing_window,
    )
