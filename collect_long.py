import os
import time
import socket
import struct
import argparse
import numpy as np
from agent.Base_Agent import Base_Agent
from behaviors.custom.Walk.Walk import Walk
from bc_data import append_dataset, sample_own_half_target
from bc_sim import FallRecovery

def reset_server_state(host="127.0.0.1", trainer_port=3101, trainer_ports=None):
    ports = (trainer_ports,) if isinstance(trainer_ports, int) else (trainer_port, 3200)
    init_cmd, kickoff_cmd, playon_cmd = b"(init (trainer))", b"(playMode BeforeKickOff)", b"(playMode PlayOn)"

    for port in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((host, port))
            s.sendall(struct.pack("!I", len(init_cmd)) + init_cmd)
            try: s.recv(1024)
            except socket.timeout: pass
            s.sendall(struct.pack("!I", len(kickoff_cmd)) + kickoff_cmd)
            time.sleep(0.05)
            s.sendall(struct.pack("!I", len(playon_cmd)) + playon_cmd)
            time.sleep(0.05)
            s.close()
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            continue
    return False

def get_long_range_target():
    target = sample_own_half_target()
    print(f"\n---> NEW LONG TARGET: [{target[0]:.2f}, {target[1]:.2f}] <---")
    return target

def save_to_disk(walk_behavior, main_file="expert_walk_data.npz"):
    temp_file = "temp_long_data.npz"
    print("\n[Disk Saver] Saving long-range dataset to disk...")
    walk_behavior.save_trajectories(temp_file)

    if os.path.exists(temp_file):
        total_samples = append_dataset(main_file, temp_file)
        print(f"[Disk Saver] Success! Total samples stored: {total_samples}")

# --- EXECUTION ---
parser = argparse.ArgumentParser(description="Collect long-range BC demonstrations")
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--dataset", default="expert_walk_data.npz")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()
if args.steps <= 0:
    parser.error("--steps must be positive")
if args.seed is not None:
    np.random.seed(args.seed)

STEPS = args.steps
DATASET_PATH = args.dataset
TEAM_NAME, UNUM = "UnifiedCollector", 1

reset_server_state(host="127.0.0.1")

agent = Base_Agent(host="localhost", agent_port=3100, monitor_port=3200, unum=UNUM, robot_type=0, team_name=TEAM_NAME)
walk_behavior = Walk(agent)
walk_behavior.is_collecting = True
fall_recovery = FallRecovery()

try:
    print(f"\n=================================================")
    print(f" SCRIPT 1: LONG-RANGE WALKING ({STEPS} STEPS)")
    print(f"=================================================")
    
    target_destination = get_long_range_target()
    force_init_step = True

    for step in range(1, STEPS + 1):
        was_reset = fall_recovery.check_and_recover(agent)
        if was_reset:
            force_init_step = True
            target_destination = get_long_range_target()

        my_pos = agent.world.robot.loc_head_position[:2]
        distance_to_target = np.linalg.norm(target_destination - my_pos)

        if distance_to_target < 0.5:
            target_destination = get_long_range_target()
            distance_to_target = np.linalg.norm(target_destination - my_pos)

        is_first = (step == 1) or force_init_step
        walk_behavior.execute(is_first, target_destination, True, None, True, distance_to_target)
        force_init_step = False

        command_string = agent.world.robot.get_command()
        agent.scom.commit_and_send(command_string if command_string else b"")
        # In synchronous mode, sending `(syn)` advances exactly one simulation
        # cycle; receive the state produced by that cycle before continuing.
        agent.scom.receive()

        if step % 1000 == 0:
            print(f"Long-Range | Step: {step}/{STEPS} | Target Dist: {distance_to_target:.2f}m")

finally:
    save_to_disk(walk_behavior, DATASET_PATH)
    print(f"[Collection Metrics] Falls recovered: {fall_recovery.fall_count}")
    print("[Script 1] Terminating connection...")
    try: agent.terminate()
    except Exception: pass
