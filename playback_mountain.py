import os 
import genome
import sys
import creature
import pybullet as p
import pybullet_data
import time 
import random
import numpy as np

def main(csv_file):
    assert os.path.exists(csv_file), "Tried to load " + csv_file + " but it does not exist"

    # 1. SETUP SIMULATION (GUI MODE)
    # We use the same physics parameters as your training simulation
    p.connect(p.GUI)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    
    # Camera setup to look at the mountain
    p.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])

    # 2. CREATE ENVIRONMENT (Exact copy of simulation.py)
    arena_size = 40
    mountain_pos_z = -1
    
    # Floor
    floor_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, 0.5])
    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, 0.5], rgbaColor=[1, 1, 0, 1])
    floor_id = p.createMultiBody(0, floor_shape, floor_vis, [0, 0, -0.5])

    # Mountain
    mountain_id = -1
    try:
        p.setAdditionalSearchPath('shapes/')
        mountain_id = p.loadURDF("gaussian_pyramid.urdf", [0, 0, mountain_pos_z], useFixedBase=1)
        
        # Apply the High Friction if you used it in training
        # If you didn't add the friction line in simulation.py yet, comment this out!
        p.changeDynamics(mountain_id, -1, lateralFriction=1.5)
        
    except:
        print("Error: Could not load mountain URDF.")

    # 3. LOAD CREATURE FROM CSV
    print(f"Loading genome from: {csv_file}")
    cr = creature.Creature(gene_count=1)
    dna = genome.Genome.from_csv(csv_file)
    cr.update_dna(dna)
    
    # Save to temp XML
    with open('playback_temp.urdf', 'w') as f:
        f.write(cr.to_xml())
        
    # 4. SPAWN CREATURE (Exact same spot as training)
    # Start Position: (0, -7, 1.0)
    start_pos = [0, -7, 1.0]
    rob1 = p.loadURDF('playback_temp.urdf', start_pos)
    
    # Apply Friction to creature (if used in training)
    for link_id in range(-1, p.getNumJoints(rob1)):
        p.changeDynamics(rob1, link_id, lateralFriction=1.5)

    # 5. RUN SIMULATION LOOP
    print("Simulation started... Press Ctrl+C to stop.")
    
    # Setup Metrics
    peak_position = np.array([0, 0, 4])
    start_dist_to_peak = np.linalg.norm(np.array(start_pos) - peak_position)
    min_dist_to_peak = start_dist_to_peak
    max_mountain_z = 0.0
    
    # Time settings
    wait_time = 1.0/240 
    total_time = 60 # Run for 60 seconds
    elapsed_time = 0
    step = 0
    
    while True:
        p.stepSimulation()
        step += 1
        
        # Motor Update (Every 24 steps = 0.1 seconds)
        if step % 24 == 0:
            motors = cr.get_motors()
            for jid in range(p.getNumJoints(rob1)):
                mode = p.VELOCITY_CONTROL
                vel = motors[jid].get_output()
                p.setJointMotorControl2(rob1, jid, controlMode=mode, targetVelocity=vel, force=5)

        # Update Metrics
        pos, orn = p.getBasePositionAndOrientation(rob1)
        x, y, z = pos
        pos_arr = np.array(pos)
        
        # Check contact with Mountain
        contact_points = p.getContactPoints(bodyA=rob1, bodyB=mountain_id)
        is_touching_mountain = len(contact_points) > 0
        
        # Update Stats
        dist_to_peak = np.linalg.norm(pos_arr - peak_position)
        if dist_to_peak < min_dist_to_peak:
            min_dist_to_peak = dist_to_peak
            
        if is_touching_mountain:
            if z > max_mountain_z:
                max_mountain_z = z
        
        # Print status every second (approx 240 steps)
        if step % 240 == 0:
            # Calculate current fitness score based on your formula
            dist_score = max(0.0, start_dist_to_peak - min_dist_to_peak)
            current_fitness = dist_score * (1.0 + max_mountain_z)
            
            print(f"Time: {elapsed_time:.1f}s | Height (on mtn): {max_mountain_z:.2f}m | Fitness: {current_fitness:.2f}")

        time.sleep(wait_time)
        elapsed_time += wait_time
        
        if elapsed_time > total_time:
            break

    print("-" * 30)
    print("Final Max Height on Mountain:", max_mountain_z)
    print("Final Fitness Score:", current_fitness)
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Usage: python playback_mountain.py elite_XX.csv")