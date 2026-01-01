import pybullet as p
import pybullet_data
import numpy as np
import math
import random
from multiprocessing import Pool

class Simulation: 
    def __init__(self, sim_id=0):
        self.physicsClientId = p.connect(p.GUI)
        self.sim_id = sim_id
        self.arena_size = 40
        self.mountain_pos_z = -1
        # Target is the center of the arena (X=0, Y=0)
        self.peak_xy = np.array([0, 0]) 

    def run_creature(self, cr, iterations=2400):
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)
        p.setGravity(0, 0, -10, physicsClientId=pid)

        # --- 1. SETUP WORLD ---
        # Floor
        floor_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.arena_size/2, self.arena_size/2, 0.5], physicsClientId=pid)
        floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.arena_size/2, self.arena_size/2, 0.5], rgbaColor=[1, 1, 0, 1], physicsClientId=pid)
        floor_id = p.createMultiBody(0, floor_shape, floor_vis, [0, 0, -0.5], physicsClientId=pid)

        # Mountain
        mountain_id = -1
        try:
            p.setAdditionalSearchPath('shapes/', physicsClientId=pid)
            mountain_id = p.loadURDF("gaussian_pyramid.urdf", [0, 0, self.mountain_pos_z], useFixedBase=1, physicsClientId=pid)
            
            # --- CRITICAL FIX: HIGH FRICTION ---
            # Increase friction to 1.5 (like rubber/rock) so they can grip the slope.
            p.changeDynamics(mountain_id, -1, lateralFriction=1.5, physicsClientId=pid)
        except:
            print("Error: Could not load mountain URDF.")

        # --- 2. SPAWN CREATURE ---
        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        xml_str = cr.to_xml()
        with open(xml_file, 'w') as f:
            f.write(xml_str)
        
        # Start 7 units away
        start_pos = [0, -7, 1.0] 
        cid = p.loadURDF(xml_file, start_pos, physicsClientId=pid)

        # Apply High Friction to the creature too
        for link_id in range(-1, p.getNumJoints(cid, physicsClientId=pid)):
            p.changeDynamics(cid, link_id, lateralFriction=1.5, physicsClientId=pid)

        # --- 3. RUN SIMULATION ---
        # We only care about Horizontal Distance (XY) to the center
        start_xy = np.array(start_pos[:2])
        start_dist_xy = np.linalg.norm(start_xy - self.peak_xy)
        min_dist_xy = start_dist_xy
        
        max_mountain_z = 0.0   
        frames_in_air = 0     
        disqualified = False

        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            
            if step % 24 == 0:
                self.update_motors(cid=cid, cr=cr)

            pos, orn = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            x, y, z = pos
            pos_xy = np.array([x, y])

            # --- A. CONTACT DETECTION ---
            contact_mountain = p.getContactPoints(bodyA=cid, bodyB=mountain_id, physicsClientId=pid)
            # We assume floor contact is safe, but we track mountain contact specifically for the bonus
            contact_floor = p.getContactPoints(bodyA=cid, bodyB=floor_id, physicsClientId=pid)
            
            is_touching_mountain = len(contact_mountain) > 0
            is_touching_ground = is_touching_mountain or (len(contact_floor) > 0)

            # --- B. AIR TIME CHECK (Permits Jumping) ---
            if is_touching_ground:
                frames_in_air = 0
            else:
                frames_in_air += 1
            
            # Allow 1 second (240 frames) of air time for jumps.
            if frames_in_air > 240:
                disqualified = True
                break

            # Bounds Check
            if (abs(x) > 19) or (abs(y) > 19) or (z < -1):
                disqualified = True
                break

            # --- C. TRACK METRICS ---
            # 1. Update XY Progress (Walking)
            dist_xy = np.linalg.norm(pos_xy - self.peak_xy)
            if dist_xy < min_dist_xy:
                min_dist_xy = dist_xy
            
            # 2. Update Z Progress (Climbing)
            # Only count height if we are PHYSICALLY GRIPPING the mountain
            if is_touching_mountain:
                if z > max_mountain_z:
                    max_mountain_z = z

        # --- 4. ASSIGN SCORE ---
        if disqualified:
            cr.fitness = 0.0
        else:
            # Base Score: HOW FAR DID YOU WALK? (XY only)
            # If they just grow tall but stay at X=0, Y=-7, this score is ZERO.
            walk_score = max(0.0, start_dist_xy - min_dist_xy)
            
            # Multiplier: HOW HIGH DID YOU CLIMB?
            # If they walk on floor: Multiplier = 1.0 (Score = Walk distance)
            # If they climb: Multiplier increases drastically.
            climb_multiplier = 1.0 + (max_mountain_z * 2.0)
            
            cr.fitness = walk_score * climb_multiplier
    
    def update_motors(self, cid, cr):
        for jid in range(p.getNumJoints(cid, physicsClientId=self.physicsClientId)):
            m = cr.get_motors()[jid]
            p.setJointMotorControl2(cid, jid, 
                    controlMode=p.VELOCITY_CONTROL, 
                    targetVelocity=m.get_output(), 
                    force = 5, 
                    physicsClientId=self.physicsClientId)

    def eval_population(self, pop, iterations):
        for cr in pop.creatures:
            self.run_creature(cr, iterations) 

# (Keep your ThreadedSim class below as normal)
class ThreadedSim():
    def __init__(self, pool_size):
        # We don't create simulations here anymore because they can't be pickled
        self.pool_size = pool_size

    @staticmethod
    def static_run_creature(sim_id, cr, iterations):
        # Create the simulation LOCALLY inside the worker process
        sim = Simulation(sim_id)
        sim.run_creature(cr, iterations)
        
        # Disconnect to clean up memory/threads
        p.disconnect(sim.physicsClientId)
        return cr
    
    def eval_population(self, pop, iterations):
        pool_args = [] 
        start_ind = 0
        pool_size = self.pool_size
        
        while start_ind < len(pop.creatures):
            this_pool_args = []
            for i in range(start_ind, start_ind + pool_size):
                if i == len(pop.creatures):
                    break
                # We pass the ID (int), not the Simulation object
                sim_ind = i % pool_size
                this_pool_args.append([
                            sim_ind, 
                            pop.creatures[i], 
                            iterations]   
                )
            pool_args.append(this_pool_args)
            start_ind = start_ind + pool_size

        new_creatures = []
        for pool_argset in pool_args:
            # use 'spawn' context to ensure thread safety on Linux
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(pool_size) as p:
                creatures = p.starmap(ThreadedSim.static_run_creature, pool_argset)
                new_creatures.extend(creatures)
        pop.creatures = new_creatures