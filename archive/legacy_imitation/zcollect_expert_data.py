import os
import time
import socket
import struct
import numpy as np
from agent.Base_Agent import Base_Agent
from behaviors.custom.Walk.Walk import Walk

# ---------------------------------------------------------
# Trainer Socket Automation (RCSSServer3D)
# ---------------------------------------------------------
def reset_server_state(host="127.0.0.1", trainer_port=3101, trainer_ports=None):
    """
    Connects to the SimSpark trainer port using explicit IPv4.
    Accepts both singular 'trainer_port' and tuple/list 'trainer_ports'.
    """
    if trainer_ports is not None:
        ports = (trainer_ports,) if isinstance(trainer_ports, int) else trainer_ports
    else:
        ports = (trainer_port, 3200)

    init_cmd = b"(init (trainer))"
    pm_cmd = b"(playMode PlayOn)"
    time_cmd = b"(time 0)"

    for port in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((host, port))
            
            s.sendall(struct.pack("!I", len(init_cmd)) + init_cmd)
            try:
                s.recv(1024)
            except socket.timeout:
                pass
                
            s.sendall(struct.pack("!I", len(pm_cmd)) + pm_cmd)
            s.sendall(struct.pack("!I", len(time_cmd)) + time_cmd)
            s.close()
            
            print(f"[Server Automation] Clock reset to 0s | playMode set to 'PlayOn' via port {port}.")
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            continue

    print(f"[Server Automation Note] Trainer port inactive on ports {ports}. Agent continuing normal execution.")
    return False
# Reset server clock and game mode prior to agent startup
reset_server_state(host="localhost", trainer_port=3101)

# 1. Initialize the agent and connect to the server
agent = Base_Agent(host="localhost", agent_port=3100, monitor_port=3200, 
                   unum=1, robot_type=0, team_name="DataCollector")

# 2. Instantiate the Walk behavior and enable collection
walk_behavior = Walk(agent)
walk_behavior.is_collecting = True

print("Starting data collection. Make sure rcssserver3d is running!")

num_steps = 10000 
target_destination = np.array([5.0, 0.0])
interrupted = False

# 3. Step the environment to collect data
try:
    for step in range(num_steps):
        # 1. Receive and parse
        msg = agent.scom.receive()
        if msg:
            agent.world.update(msg) 
            
        # 2. Calculate distance and handle continuous random waypoints
        my_pos = agent.world.robot.loc_head_position[:2]
        distance_to_target = np.linalg.norm(target_destination - my_pos)
        
        if distance_to_target < 0.5:
            target_destination = np.random.uniform(-8.0, 8.0, size=2)
            print(f"*** Target reached! Redirecting to: {target_destination} ***")
            distance_to_target = np.linalg.norm(target_destination - my_pos)
        
        # 3. Command the expert
        is_reset = (step == 0)
        walk_behavior.execute(is_reset, target_destination, True, None, True, distance_to_target)
        
        # 4. Get the formatted string of joint commands
        command_string = agent.world.robot.get_command()
        
        # --- DIAGNOSTIC PRINTS (Every 100 steps) ---
        if step % 100 == 0:
            play_mode = getattr(agent.world, 'play_mode', "Unknown")
            print(f"Step: {step}/{num_steps} | Target Dist: {distance_to_target:.2f}m | PlayMode: {play_mode}")
        # -------------------------------------------
        
        # 5. Send actions back to the server
        if command_string:
            agent.scom.commit_and_send(command_string)
        else:
            agent.scom.commit_and_send(b"" if isinstance(msg, bytes) else "")

except KeyboardInterrupt:
    print("\nCollection interrupted by user. Discarding gathered steps...")
    interrupted = True

# ---------------------------------------------------------
# 4. DATASET APPENDING LOGIC (Only runs on full completion)
# ---------------------------------------------------------
if not interrupted:
    temp_file = "ztemp_expert_data.npz"
    main_file = "zexpert_walk_data.npz"

    print("Extracting trajectories from agent memory...")
    walk_behavior.save_trajectories(temp_file)

    if os.path.exists(main_file):
        print(f"Found existing dataset at '{main_file}'. Merging data...")
        old_data = np.load(main_file)
        new_data = np.load(temp_file)
        
        combined_data = {}
        for key in new_data.files:
            if key in old_data:
                combined_data[key] = np.concatenate((old_data[key], new_data[key]), axis=0)
            else:
                combined_data[key] = new_data[key]
                
        np.savez_compressed(main_file, **combined_data)
        os.remove(temp_file)
        
        first_key = list(combined_data.keys())[0]
        total_steps = len(combined_data[first_key])
        print(f"Successfully appended! Total dataset size is now: {total_steps} steps.")
        
    else:
        print(f"No existing dataset found. Creating '{main_file}'...")
        os.rename(temp_file, main_file)

    print("Data collection session complete.")