# If you on a Windows machine with any Python version 
# or an M1 mac with any Python version
# or an Intel Mac with Python > 3.7
# this multi-threaded version does not work
# please use test_ga_single_thread.py on those setups

import unittest
import population
import simulation 
import genome 
import creature 
import numpy as np

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        # 1. SETUP
        # Initial population with 3 genes (segments)
        pop = population.Population(pop_size=30, 
                                    gene_count=3)
        
        # Initialize the Mountain Simulation
        # pool_size=1 is safe for all machines (Windows/Mac)
        sim = simulation.Simulation()

        # 2. GENERATIONAL LOOP
        for iteration in range(1000):
            
            # Run the simulation for this generation
            # 2400 steps = 10 seconds of simulation time
            sim.eval_population(pop, 2400)
            
            # --- UPDATED: Use cr.fitness (Mountain Score) ---
            fits = [cr.fitness for cr in pop.creatures]
            
            # Track complexity (number of links)
            links = [len(cr.get_expanded_links()) for cr in pop.creatures]
            
            # Print stats
            print(f"Gen {iteration} | Fittest: {np.max(fits):.3f} | Mean: {np.mean(fits):.3f} | Mean Links: {np.mean(links):.1f}")       
            
            # 3. REPRODUCTION (Genetic Algorithm)
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            
            for i in range(len(pop.creatures)):
                # Select Parents
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                
                # Crossover & Mutate
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.05)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)
                
                # Create Child
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)
            
            # 4. ELITISM (Save the King of the Mountain)
            # We keep the best creature from the previous generation unchanged
            max_fit = np.max(fits)
            for cr in pop.creatures:
                if cr.fitness == max_fit:
                    new_cr = creature.Creature(1)
                    new_cr.update_dna(cr.dna)
                    new_creatures[0] = new_cr
                    
                    # Save the Elite DNA to file
                    filename = "csv/elite_"+str(iteration)+".csv"
                    genome.Genome.to_csv(cr.dna, filename)
                    break
            
            # Update population for next round
            pop.creatures = new_creatures
                            
        self.assertNotEqual(fits[0], 0)

unittest.main()