import os
import time
import numpy as np
import math
from agent.Base_Agent import Base_Agent
from behaviors.custom.Walk.Walk import Walk

# ==========================================
# 1. Initialize the Expert Agent
# ==========================================
agent = Base_Agent(host="localhost", agent_port=3100, monitor_port=3200, 
                   unum=1, robot_type=0, team_name="ShortTurnCollector")

walk_behavior = Walk(agent)
walk_behavior.is_collecting = True  # Enable data saving mode

print("Starting short-range data collection...")
num_steps = 5000  # Increased steps to gather plenty of turning data

# ==========================================
# 2. Short-Range Target Generator
# ==========================================
def get_short_range_target(current_pos):
    """
    Generates a new target 1.0 to 2.0 meters away in a random direction,
    forcing constant turning and pivoting.
    """
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(1.0, 2.0)
    
    tx = current_pos[0] + np.cos(angle) * dist
    ty = current_pos[1] + np.sin(angle) * dist
    
    # Keep the robot from wandering entirely off the field
    tx = np.clip(tx, -6.0, 6.0)
    ty = np.clip(ty, -6.0, 6.0)
    
    print(f"\n---> NEW SHORT TARGET: [{tx:.2f}, {ty:.2f}] (Dist: {dist:.2f}m) <---")
    return np.array([tx, ty])

# Initialize dummy target to get started
target_destination = np.array([1.0, 0.0]) 

# ==========================================
# 3. The Simulation Loop
# ==========================================
interrupted = False

try:
    for step in range(num_steps):
        msg = agent.scom.receive()
        if msg: 
            agent.world.update(msg) 
            
        # Get ground-truth global position
        my_pos = agent.world.robot.loc_head_position[:2]
        
        # Calculate distance to current target
        distance_to_target = np.linalg.norm(target_destination - my_pos)
        
        # Tighter threshold (0.4m) so it makes sharper pivots upon arrival
        if distance_to_target < 0.4 or step == 0:
            target_destination = get_short_range_target(my_pos)
            distance_to_target = np.linalg.norm(target_destination - my_pos)
        
        # Execute expert behavior
        is_reset = (step == 0)
        walk_behavior.execute(is_reset, target_destination, True, None, True, distance_to_target)
        
        # Send commands
        command_string = agent.world.robot.get_command()
        
        if step % 100 == 0:
            print(f"Step: {step}/{num_steps} | Target Dist: {distance_to_target:.2f}m")
        
        agent.scom.commit_and_send(command_string if command_string else b"")

except KeyboardInterrupt:
    print("\nCollection interrupted by user. Discarding collected data...")
    interrupted = True

# ==========================================
# 4. Save and Append Dataset (Only if not interrupted)
# ==========================================
if not interrupted:
    temp_file = "temp_short_data.npz"
    main_file = "expert_walk_data.npz"

    print("\nProcessing collected trajectories...")
    walk_behavior.save_trajectories(temp_file)

    if os.path.exists(main_file):
        print(f"Appending new short-range data to existing '{main_file}'...")
        old_data = np.load(main_file)
        new_data = np.load(temp_file)
        
        # Concatenate old and new data
        combined_data = {
            k: np.concatenate((old_data[k], new_data[k]), axis=0) if k in old_data else new_data[k] 
            for k in new_data.files
        }
        np.savez_compressed(main_file, **combined_data)
        os.remove(temp_file)
    else:
        print(f"No existing dataset found. Creating '{main_file}'...")
        os.rename(temp_file, main_file)

    total_samples = np.load(main_file)['obs'].shape[0]
    print(f"Data collection complete! Dataset now contains {total_samples} total state-action pairs.")