import numpy as np

# generic pso for values
def pso(
        cost_func, 
        dim=3, 
        num_particles=10, 
        max_iter=100, 
        w=0.5, 
        c1=1, 
        c2=2, 
        input=None, 
        y=None, 
        baselines=[], 
        override=False, 
        override_particles=None,
        lrp_type="gamma"):
    # Initialize particles and velocities
    particles = np.random.uniform(0, 1, (num_particles, dim))
    # constrain to 1
    if dim != 1:
        particles[:, 0] = np.random.uniform(1, 2, num_particles) # alpha
        particles[:, 1] = np.random.uniform(0.0000005, 0.00000005, num_particles) # epsilon

    velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    
    if override:
        print("****** overriding values ******")
        particles = override_particles
        
        best_positions = np.copy(particles)
        # manual override
        best_fitness = np.array([-0.5724872,  -0.83166267,  0.21900438,  0.86542261, -0.3303999 ])
        swarm_best_position = np.array([1.00000000e+00, 7.99812145e-08, 1.07015182e+00])
        swarm_best_fitness = -1.2439818322406806
    else:
        best_positions = np.copy(particles)
        best_fitness = np.array([cost_func(p, y, input, baselines) for p in particles])
        swarm_best_position = best_positions[np.argmin(best_fitness)]
        swarm_best_fitness = np.min(best_fitness)

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        print("r1:", r1)
        print("r2:", r2)
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities
        if (dim == 1 and lrp_type == "alpha") or dim > 1:
            particles[:, 0] = np.clip(particles[:, 0], 1, 3) # alpha
        
        fitness_values = np.array([cost_func(p, y, input, baselines) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)

        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)
        print("particles: ", particles, "best position: ",  swarm_best_position)
        print("fitness: ", fitness_values, "best fitness: ",  swarm_best_fitness)

    # Print the solution and fitness value
    print('Solution:', swarm_best_position)
    print('Fitness:', swarm_best_fitness)
    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness

# discretized pso for layers
def pso_layers(cost_func, dim=3, num_particles=10, max_iter=100, w=0.5, c1=1, c2=2, input=None, y=None, baselines=[], override=False, override_particles=None, override_fitness=None):
    # Initialize particles and velocities
    particles = np.random.uniform(0, 4, (num_particles, dim)).astype(float)

    velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    if override:
        particles = override_particles 
        best_positions = np.copy(particles)
        best_fitness = override_fitness 
        swarm_best_position = np.array([3.0, 3.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 3.0, 3.0])
        swarm_best_fitness = -1.890534217657596
    else:
        best_positions = np.copy(particles)
        best_fitness = np.array([cost_func(p, y, input, baselines) for p in particles])
        swarm_best_position = best_positions[np.argmin(best_fitness)]
        swarm_best_fitness = np.min(best_fitness)

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1.5, (num_particles, dim))
        r2 = np.random.uniform(0, 1.5, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities
        particles = np.floor(particles)
        particles = np.clip(particles, 0, 3)

        fitness_values = np.array([cost_func(p, y, input, baselines) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)
        print("particles: ", particles, "best position: ",  swarm_best_position)
        print("fitness: ", fitness_values, "best fitness: ",  swarm_best_fitness)

    # Print the solution and fitness value
    print('Solution:', swarm_best_position)
    print('Fitness:', swarm_best_fitness)
    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness
