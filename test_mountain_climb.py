import pybullet as p
import pybullet_data
import time
import numpy as np
import creature
import math

# ---------------------------------------------------------
# 1. SETUP ENVIRONMENT
# ---------------------------------------------------------

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

def make_arena(arena_size=20, wall_height=1):
    wall_thickness = 0.5
    floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness])
    floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, -wall_thickness])
    # Walls
    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size/2, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size/2, wall_height/2])
    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[arena_size/2, 0, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[-arena_size/2, 0, wall_height/2])

arena_size = 40
make_arena(arena_size=arena_size)

# Mountain Setup
p.setAdditionalSearchPath('shapes/')
mountain_pos_z = -1
try:
    mountain = p.loadURDF("gaussian_pyramid.urdf", (0, 0, mountain_pos_z), (0, 0, 0, 1), useFixedBase=1)
except:
    print("Error loading mountain.")

# ---------------------------------------------------------
# 2. HELPER: TERRAIN HEIGHT CALCULATOR
# ---------------------------------------------------------
def get_local_terrain_height(x, y):
    """
    Calculates the expected Z height of the ground at (x,y).
    Logic: The ground is max(floor_level, mountain_level).
    Mountain is Gaussian(height=5, sigma=3) shifted down by 1 unit.
    """
    floor_level = 0
    
    # These params must match your prepare_shapes.py
    sigma = 3
    peak_height = 5
    
    # Calculate mountain shape
    mountain_z = peak_height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))
    
    # Apply the offset (we placed the mountain at z = -1)
    actual_mountain_z = mountain_z + mountain_pos_z
    
    return max(floor_level, actual_mountain_z)

# ---------------------------------------------------------
# 3. SPAWN CREATURE
# ---------------------------------------------------------
cr = creature.Creature(gene_count=3)
with open('test_mountain_creature.urdf', 'w') as f:
    f.write(cr.to_xml())

start_pos = [0, -8, 0.5] # Start low so we don't trigger "flying" immediately
creature_id = p.loadURDF('test_mountain_creature.urdf', start_pos)
motors = cr.get_motors()

# ---------------------------------------------------------
# 4. SIMULATION WITH STRICT CHECK
# ---------------------------------------------------------
print("Simulation started...")
p.setRealTimeSimulation(0)

peak_position = np.array([0, 0, 4]) 
simulation_steps = 10000000000000

# Get initial stats
start_pos_vec, _ = p.getBasePositionAndOrientation(creature_id)
start_dist_to_peak = np.linalg.norm(np.array(start_pos_vec) - peak_position)
min_dist_to_peak = start_dist_to_peak
disqualified = False

for step in range(simulation_steps):
    p.stepSimulation()
    
    # Motor Control
    for jid in range(p.getNumJoints(creature_id)):
        if jid < len(motors):
            p.setJointMotorControl2(creature_id, jid, p.VELOCITY_CONTROL, 
                                    targetVelocity=motors[jid].get_output(), force=5)
    
    # Position Update
    pos, _ = p.getBasePositionAndOrientation(creature_id)
    x, y, z = pos

    # --- STRICT FLIGHT CHECK ---
    ground_z = get_local_terrain_height(x, y)
    fly_threshold = 1.5 
    
    if z > (ground_z + fly_threshold):
        print(f"\n!!! DISQUALIFIED: Flying Detected at step {step} !!!")
        print(f"Creature Z: {z:.2f} | Ground Z here: {ground_z:.2f}")
        disqualified = True
        break
        
    # --- Bounds Check ---
    if (abs(x) > 19) or (abs(y) > 19) or (z < -1):
        print(f"\n!!! DISQUALIFIED: Out of Bounds !!!")
        disqualified = True
        break

    # Fitness Tracking
    dist_to_peak = np.linalg.norm(np.array(pos) - peak_position)
    if dist_to_peak < min_dist_to_peak:
        min_dist_to_peak = dist_to_peak

    # --- DEBUG PRINT (Add this back!) ---
    # Print every 10 steps so you can see it moving
    if step % 10 == 0:
        print(f"Step {step}: Current Dist: {dist_to_peak:.2f} | Best Dist: {min_dist_to_peak:.2f}")

    time.sleep(1/240)
# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
print("-" * 30)
if disqualified:
    fitness = -10.0
    print("Result: DISQUALIFIED (Flying/Cheating)")
else:
    fitness = start_dist_to_peak - min_dist_to_peak
    if fitness < 0: fitness = 0
    print(f"Start Dist: {start_dist_to_peak:.2f}")
    print(f"Best Dist:  {min_dist_to_peak:.2f}")
    print(f"FITNESS:    {fitness:.4f}")
print("-" * 30)

time.sleep(2)
p.disconnect()