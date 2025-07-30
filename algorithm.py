import numpy as np
from particle import Particle
from collections import Counter

params = {
    "max_cpu": 0.95,  # Maximum CPU usage of the cluster   
    "max_ram": 0.90,  # Maximum RAM usage of the cluster
    "w_max": 0.7,     # Maximum weight of the particle
    "w_min": 0.5,     # Minimum weight of the particle
    "c1_min": 0.01,   # Minimum learning factor (C1) of the particle
    "c1_max": 2.5,    # Maximum learning factor of the (C1) particle
    "c2_min": 0.01,   # Minimum learning factor (C2) of the particle
    "c2_max": 2.5,    # Maximum learning factor (C2) of the particle
    "weights": [0.4, 0.5, 0.10],  # Weights of the fitness function
    "theta": 3.4,     # weight control factor
    "max_itr": 120,   # Maximum number of iterations
    "tmax": 100       # Maximum time for calculating C1 and C2
}

class PSO:
    """
    Particle Swarm Optimization (PSO) class for resource allocation and scheduling.
    Attributes:
        particles (list): List of Particle objects representing the swarm.
        global_best_position (np.ndarray): Best positions found by the swarm.
        global_best_fitness (np.ndarray): Best fitness values found by the swarm.
        best_fitness (float): Best overall fitness found.
        c1 (float): Cognitive learning factor.
        c2 (float): Social learning factor.
        w (float): Inertia weight.
        w_max (float): Maximum inertia weight.
        w_min (float): Minimum inertia weight.
        c1_min (float): Minimum cognitive learning factor.
        c1_max (float): Maximum cognitive learning factor.
        c2_min (float): Minimum social learning factor.
        c2_max (float): Maximum social learning factor.
        weights (tuple): Weights for reward and penalty in fitness calculation.
        theta (float): Parameter for inertia weight calculation.
        max_cpu (float): Maximum allowed CPU usage proportion.
        max_ram (float): Maximum allowed RAM usage proportion.
        max_itr (int): Maximum number of iterations.
        tmax (int): Maximum time or iteration count.
    Methods:
        calculate_weigth(itr):
            Updates the inertia weight based on the current iteration.
        calculate_learning_factors(itr):
            Updates the cognitive and social learning factors based on the current iteration.
        greedy_initialisation(clusters, components, component_id):
            Returns feasible cluster IDs for a component based on resource availability.
        calculate_scores(feasible_positions, component, clusters, itr):
            Calculates and returns normalized scores for feasible cluster positions.
        initialise_swarm(num_particles, clusters, components):
            Initializes the swarm with feasible positions for each particle/component.
        is_feasible(clusters, components, use_current_position):
            Checks if the current or best positions of the swarm are feasible given resource constraints.
        access_fitness(components, clusters, cluster_id, component_id, feasibility):
            Computes the fitness value for a given component placement.
        iteration(clusters, components, population_size):
            Runs the PSO optimization for a given number of iterations and returns the best solution found.
        scheduler(clusters, components):
            Runs multiple PSO iterations to find the best scheduling solution and its fitness.
    """
    def __init__(self) -> None:
        self.particles = []
        self.global_best_position = np.array([])
        self.global_best_fitness = np.array([])
        self.best_fitness = 0
        self.c1 = 0
        self.c2 = 0
        self.w = 0
        
        self.w_max = params['w_max']
        self.w_min = params['w_min']
        self.c1_min = params['c1_min']
        self.c1_max = params['c1_max']
        self.c2_min = params['c2_min']
        self.c2_max = params['c2_max']
        self.weights = params['weights']
        self.theta = params['theta']
        self.max_cpu = params['max_cpu']
        self.max_ram = params['max_ram']
        self.max_itr = params['max_itr']
        self.tmax = params['tmax']

    def calculate_weigth(self, itr: int):
        """
        Calculates and updates the weight parameter `self.w` based on the current iteration.

        The weight is computed using an exponential decay formula that depends on the current iteration (`itr`),
        the maximum number of iterations (`self.tmax`), and the parameters `self.theta`, `self.w_min`, and `self.w_max`.

        Args:
            itr (int): The current iteration number.

        Updates:
            self.w (float): The updated weight value.
        """

        # Calculate the auxiliary term for exponential decay
        Waux = (self.theta * np.square(itr) / np.square(self.tmax))
        # Calculate the difference between max and min weights
        Wdiff = self.w_max - self.w_min
        # Apply the exponential decay formula to determine the current weight
        self.w = self.w_min + (Wdiff * np.exp(-Waux))

    def calculate_learning_factors(self, itr: int):
        """
        Calculates and updates the learning factors c1 and c2 based on the current iteration.
        The learning factors are computed using a quadratic decay formula, where their values decrease
        from their respective maximums (c1_max, c2_max) to minimums (c1_min, c2_min) as the iteration
        number increases from 0 to tmax.
        Args:
            itr (int): The current iteration number.
        Updates:
            self.c1 (float): Updated cognitive learning factor.
            self.c2 (float): Updated social learning factor.
        """

       
        squared_tmax = np.square(self.tmax) # Pre-calculate squared tmax for efficiency
        squared_itr = np.square(itr) # Pre-calculate squared iteration for efficiency

        # Calculate the rate of change for c1 and c2
        c1_aux = ((self.c1_max - self.c1_min)/squared_tmax)
        c2_aux = ((self.c2_max - self.c2_min)/squared_tmax)

        # Update c1 and c2 using the quadratic decay formula
        self.c1 = self.c1_max - (c1_aux*squared_itr)
        self.c2 = self.c2_max - (c2_aux*squared_itr)    

    def greedy_initialisation(self, clusters: dict, components: dict, component_id: int) -> np.ndarray:
        """
        Determines feasible clusters for deploying a given component using a greedy approach.
        This method calculates the total resource requirements (RAM, CPU, Disk) for a specified component,
        based on its number of replicas and per-replica resource needs. It then identifies clusters that
        have sufficient available resources to host all replicas of the component.
        Args:
            clusters (dict): A dictionary containing arrays of available resources for each cluster.
                Expected keys: 'available_ram', 'available_cpu', 'available_dsk'.
            components (dict): A dictionary containing arrays of component specifications.
                Expected keys: 'replicas', 'ram', 'cpu', 'dsk'.
            component_id (int): The 1-based index of the component to be placed.
        Returns:
            np.ndarray: An array of 1-based cluster IDs where the component can be feasibly deployed.
        """
        
        component_idx = component_id - 1  # Convert 1-based component_id to 0-based index
        
        # Calculate total resource requirements for the component based on its replicas
        required_ram = components['replicas'][component_idx] * components['ram'][component_idx]
        required_cpu = components['replicas'][component_idx] * components['cpu'][component_idx]
        required_disk = components['replicas'][component_idx] * components['dsk'][component_idx]

        # Create a boolean mask indicating which clusters have sufficient resources
        feasible_mask = (
            (clusters['available_ram'] >= required_ram) &
            (clusters['available_cpu'] >= required_cpu) &
            (clusters['available_dsk'] >= required_disk)
        )
        
        # Return the 1-based IDs of the feasible clusters
        return np.where(feasible_mask)[0] + 1 

    def calculate_scores(self, feasible_positions: np.ndarray, component: dict, clusters: dict, itr: int) -> np.ndarray:
        """
        Calculate selection probabilities for feasible cluster positions based on resource utilization and co-location constraints.
        Args:
            feasible_positions (np.ndarray): Array of feasible cluster indices (1-based).
            component (dict): Dictionary representing the component to be placed, containing at least the 'CONS' key for co-location constraints.
            clusters (dict): Dictionary containing cluster resource information with keys 'total_cpu', 'total_ram', and 'total_dsk'.
            itr (int): Current iteration or component index, used to determine which dependencies have already been placed.
        Returns:
            np.ndarray: Array of probabilities (summing to 1) for each feasible position, reflecting the desirability of each cluster based on resource usage and co-location.
        """
        scores = np.zeros(len(feasible_positions))
        
        # Get 0-based indices for feasible clusters
        feasible_indices = feasible_positions - 1 # Convert 1-based component_id to 0-based index
        
        # Calculate normalized resource utilization scores for feasible clusters
        cpu_norm = clusters['total_cpu'][feasible_indices] / clusters['total_cpu'].max()
        ram_norm = clusters['total_ram'][feasible_indices] / clusters['total_ram'].max()
        dsk_norm = clusters['total_dsk'][feasible_indices] / clusters['total_dsk'].max()
        
        # Combine resource scores (simple average)
        resource_score = (cpu_norm + ram_norm + dsk_norm) / 3

        # If there are no co-location constraints, the score is just the resource score
        if not component['CONS']: 
            scores = resource_score
        else:
            # Count how many of the already placed dependent components are in each cluster
            # Only consider components that have already been initialized (i.e., component_id < itr)
            constraints = component['CONS']
            position_counts = Counter(self.particles[ms-1].get_position() 
                         for ms in constraints if ms < itr)

            # For each feasible cluster, calculate a combined score
            for i, cluster_id in enumerate(feasible_positions):
                # Get the count of colocated dependent components for the current cluster
                colocated = position_counts.get(cluster_id, 0)
                # Normalize the colocated count by the total number of constraints
                colocated_score = colocated / len(constraints)
                # Combine colocated score and resource score with equal weights (0.5 each)
                scores[i] = 0.5 * colocated_score + 0.5 * resource_score[i]

        # Apply Softmax normalization to convert scores into probabilities
        # This ensures that positions with higher scores are more likely to be chosen,
        # and the sum of probabilities for all feasible positions is 1.
        scores = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
        return scores
    
    def initialise_swarm(self, num_particles: int, clusters: dict, components: dict) -> tuple[bool, int | None]:
        """
        Initializes the swarm of particles for a particle swarm optimization (PSO) algorithm by assigning each component to a feasible cluster based on available resources.
        Args:
            num_particles (int): The number of particles/components to initialize in the swarm.
            clusters (dict): A dictionary containing cluster resource information, including 'available_ram', 'available_cpu', and 'available_dsk'.
            components (dict): A dictionary containing component requirements, including 'CONS', 'replicas', 'ram', 'cpu', and 'dsk'.
        Returns:
            tuple[bool, int | None]: 
                - (True, None) if initialization is successful for all particles.
                - (False, component_id) if initialization fails for a specific component (returns the 1-based index of the failed component).
        Notes:
            - Each particle is assigned to a feasible cluster based on a greedy initialization and probabilistic selection.
            - Updates available resources in clusters as components are assigned.
            - Initializes each particle's position, velocity, and best-known position.
            - Sets the global best fitness and position after initialization.
        """
        # Create copies of available resources to simulate placement during initialization
        available_ram = clusters['available_ram'].copy()
        available_cpu = clusters['available_cpu'].copy()
        available_dsk = clusters['available_dsk'].copy()
        
         # Iterate through each component to initialize its corresponding particle
        for component_id in range(1, num_particles + 1):  
            particle = Particle()
            
            # Find feasible clusters for the current component based on available resources
            feasible_positions = self.greedy_initialisation(clusters, components, component_id)
            
            # If no feasible positions are found for a component, initialization fails
            if len(feasible_positions) == 0:
                return False, component_id
            
            # Extract details of the current component
            component = {
                'CONS': components['CONS'][component_id-1],
                'replicas': components['replicas'][component_id-1],
                'ram': components['ram'][component_id-1],
                'cpu': components['cpu'][component_id-1],
                'dsk': components['dsk'][component_id-1]
            }


            # Calculate scores (probabilities) for each feasible position
            probas = self.calculate_scores(feasible_positions, component, clusters, component_id)
            
            # Choose a position for the particle based on the calculated probabilities
            chosen_idx = np.random.choice(len(feasible_positions), p=probas)
            chosen_position = feasible_positions[chosen_idx]
            
            # Update the particle's position, velocity, and best position
            particle.position = chosen_position
            particle.set_velocity(1)
            particle.set_best_position(chosen_position)
            
            # Update the available resources in the chosen cluster
            cluster_idx = chosen_position - 1
            available_ram[cluster_idx] -= component['replicas'] * component['ram']
            available_cpu[cluster_idx] -= component['replicas'] * component['cpu']
            available_dsk[cluster_idx] -= component['replicas'] * component['dsk']
            
            self.particles.append(particle)
        
        # Initialize global bests after all particles have been placed
        self.global_best_fitness = np.zeros(num_particles)
        self.global_best_position = np.array([p.position for p in self.particles])
        
        # Calculate initial fitness for each particle and set its personal best and global best fitness
        for i, particle in enumerate(self.particles):
            component_id = i + 1
            current_fitness = self.access_fitness(components, clusters, particle.position, component_id, True)
            particle.update_fitness(current_fitness)
            self.global_best_fitness[i] = current_fitness

        return True, None # Initialization successful
    
    def is_feasible(self, clusters: dict, components: dict, use_current_position: bool) -> bool:
        """
        Checks whether the current or best assignment of components to clusters is feasible based on resource constraints.
        Args:
            clusters (dict): A dictionary containing available resources for each cluster. 
                Expected keys: 'available_ram', 'available_cpu', 'available_dsk', each mapping to a list or array of resource values per cluster.
            components (dict): A dictionary containing resource requirements for each component.
                Expected keys: 'replicas', 'ram', 'cpu', 'dsk', each mapping to a list or array of values per component.
            use_current_position (bool): If True, checks feasibility using the current position of each particle; 
                if False, uses the best known position.
        Returns:
            bool: True if all components can be assigned to clusters without exceeding any resource limits; False otherwise.
        """

        # Create temporary copies of available resources to avoid modifying original cluster data
        available_ram = clusters['available_ram'].copy()
        available_cpu = clusters['available_cpu'].copy()
        available_dsk = clusters['available_dsk'].copy()
        
        # Iterate through each particle (component)
        for i, particle in enumerate(self.particles):
            component_idx = i  # 0-based index for component data

            # Get the cluster ID based on whether current or best position is being checked
            cluster_id = particle.position if use_current_position else particle.best_position
            cluster_idx = cluster_id - 1

            # Subtract the resources required by the component from the chosen cluster's available resources
            available_ram[cluster_idx] -= components['replicas'][component_idx] * components['ram'][component_idx]
            available_cpu[cluster_idx] -= components['replicas'][component_idx] * components['cpu'][component_idx]
            available_dsk[cluster_idx] -= components['replicas'][component_idx] * components['dsk'][component_idx]

            # If any resource goes below zero, the configuration is infeasible
            if (available_ram < 0).any() or (available_cpu < 0).any() or (available_dsk < 0).any():
                return False
        
        return True # All components can be placed without exceeding resource limits
    
    def access_fitness(self, components: dict, clusters: dict, cluster_id: int, component_id: int, feasibility: bool) -> float:
        """
        Calculates the fitness score for assigning a specific component to a cluster, considering resource usage, 
        constraint satisfaction, and feasibility.
        Args:
            components (dict): Dictionary containing component properties such as 'replicas', 'ram', 'cpu', 'dsk', and 'CONS'.
            clusters (dict): Dictionary containing cluster properties such as 'total_ram', 'total_cpu', 'total_dsk', 
                             'available_ram', and 'available_cpu'.
            cluster_id (int): 1-based index of the target cluster.
            component_id (int): 1-based index of the component to be placed.
            feasibility (bool): Indicates whether the current swarm configuration is feasible.
        Returns:
            float: The computed fitness value. Returns a very low value (-inf) if the configuration is infeasible.
        Notes:
            - The fitness function rewards balanced resource usage (RAM, CPU, Disk) and co-location of constrained components.
            - Penalties are applied for nearing resource limits (RAM, CPU).
            - The final fitness is a weighted sum of rewards and penalties, with weights defined in self.weights.
        """
        if not feasibility:
            return -float('inf') # Return a very low fitness if the overall swarm configuration is infeasible
        
        # Unpack fitness function weights
        w_reward1, w_reward2, w_penalty = self.weights
        component_idx = component_id - 1 # Convert to 0-based index
        cluster_idx = cluster_id - 1 # Convert to 0-based index
        
        # Calculate proportion of resources used by the component within the target cluster
        ram_usage = components['replicas'][component_idx] * components['ram'][component_idx] / clusters['total_ram'][cluster_idx]
        cpu_usage = components['replicas'][component_idx] * components['cpu'][component_idx] / clusters['total_cpu'][cluster_idx]
        
        # Calculate remaining available resources after placing this component 
        remaining_ram = clusters['available_ram'][cluster_idx] - ram_usage
        remaining_cpu = clusters['available_cpu'][cluster_idx] - cpu_usage
        
        # Calculate the ram, cpu and disk usage proportion
        ram_proportion = components['replicas'][component_idx] * components['ram'][component_idx] / clusters['total_ram'][cluster_idx]
        cpu_proportion = components['replicas'][component_idx] * components['cpu'][component_idx] / clusters['total_cpu'][cluster_idx]
        dsk_proportion = components['replicas'][component_idx] * components['dsk'][component_idx] / clusters['total_dsk'][cluster_idx]
        
        # Reward 1: Balance of resource usage for the current component in the cluster
        # Measures how balanced the RAM, CPU, and Disk usage proportions are. Closer to 1 is better.
        reward = 1 - (abs(ram_proportion - cpu_proportion) + abs(ram_proportion - dsk_proportion) + abs(cpu_proportion - dsk_proportion))/3
        
        penalty = 0.0
        
        # Penalty for nearing RAM limits
        ram_threshold = clusters['total_ram'][cluster_idx] * (1 - self.max_ram)
        if remaining_ram < ram_threshold:
            penalty -= (ram_threshold - remaining_ram)/clusters['total_ram'][cluster_idx]
        
        # Penalty for nearing CPU limits
        cpu_threshold = clusters['total_cpu'][cluster_idx] * (1 - self.max_cpu)
        if remaining_cpu < cpu_threshold:
            penalty -= (cpu_threshold - remaining_cpu)/clusters['total_cpu'][cluster_idx]
        
        # Reward 2: Co-location constraint satisfaction
        # Checks how many constrained microservices (components) are already placed in the same cluster.
        constraints = components['CONS'][component_idx]
        if len(constraints) == 0:
            l_reward = 0.0
        else:
            # Count how many of the constrained microservices are currently in the same cluster
            l_reward = sum(1 for microservice in constraints if self.particles[microservice-1].get_position() == cluster_id)

        # Calculate final fitness using weighted sum
        fitness = 1 + (w_reward1 * reward) + (w_reward2 * l_reward) + (w_penalty * penalty)
        
        return fitness
    
    def iteration(self, clusters: dict, components: dict, population_size: int):
        """
        Performs a single Particle Swarm Optimization (PSO) iteration for component-to-cluster assignment.
        This method initializes the swarm, iteratively updates particle positions and velocities, evaluates fitness,
        and tracks the best solutions found. It is designed to optimize the assignment of components to clusters
        based on resource constraints and fitness evaluation.
        Args:
            clusters (dict): Dictionary containing cluster information, including available resources and IDs.
            components (dict): Dictionary containing component information and requirements.
            population_size (int): Number of particles (candidate solutions) in the swarm.
        Returns:
            tuple:
                - np.ndarray: Final positions of all particles, representing component-to-cluster assignments.
                - float: Mean of the global best fitness values across all components.
        Notes:
            - If swarm initialization fails (e.g., due to insufficient resources for any component), returns (False, 0).
            - The method updates inertia weight and learning factors dynamically during iterations.
            - Fitness is evaluated for each particle, and both personal and global bests are updated accordingly.
        """
        self.particles = [] # Clear particles from previous runs

        # Initialize the swarm. If initialization fails (e.g., no feasible cluster for a component),
        # return False and 0 fitness.
        loop_condition, failed_component = self.initialise_swarm(population_size, clusters, components)
        if not loop_condition:
            print(f"Initialization failed at component {failed_component}. Not enough resources available.")
            return False, 0
        
        # Main PSO iteration loop
        for itr in range(self.max_itr):
            self.calculate_weigth(itr)  # Update inertia weight
            self.calculate_learning_factors(itr) # Update cognitive and social learning factors

            # Check feasibility of the current swarm configuration. This determines if penalties apply.
            feasibility = self.is_feasible(clusters, components, True)
            
            # Update each particle
            for i, particle in enumerate(self.particles):
                component_id = i + 1 # 1-based component ID

                # Update particle's velocity based on its personal best and global best position
                particle.update_velocity(self.global_best_position[i], self.w, self.c1, self.c2)
                # Update particle's position, ensuring it stays within cluster ID bounds
                particle.update_position(1, len(clusters['cluster_id']))
                
                # Calculate the fitness of the particle's new position
                fitness = self.access_fitness(components, clusters, particle.position, component_id, feasibility)
                
                # Update the particle's personal best fitness and position if current fitness is better
                particle.update_fitness(fitness)
                
                # Update the global best fitness and position if the particle's current fitness is better
                # than the previously recorded global best for that component
                if fitness > self.global_best_fitness[i]:
                    self.global_best_fitness[i] = fitness
                    self.global_best_position[i] = particle.position
        
        # After all iterations, get the final best positions of all particles
        position = np.array([p.position for p in self.particles])
        # The overall best fitness of this run is the mean of the global best fitnesses for all components
        best_fitness = np.mean(self.global_best_fitness)
        
        return position, best_fitness

    def scheduler(self, clusters: dict, components: dict):
        """
        Executes a Particle Swarm Optimization (PSO) algorithm to assign components (e.g., microservices)
        to clusters, aiming to maximize a fitness function.
        The method runs the PSO process multiple times with different initializations to increase the
        likelihood of finding a global optimum. For each run, it updates the best solution found if a
        better fitness value is achieved.
        Args:
            clusters (dict): A dictionary representing the available clusters/resources.
            components (dict): A dictionary containing the components (e.g., microservices) to be assigned,
                including a key 'ms_i' representing the list of components.
        Returns:
            tuple: A tuple containing:
                - solution: The best cluster assignment found (format depends on the implementation of `iteration`).
                - best_fitness (float): The fitness value of the best solution found.
        """
        self.best_fitness = 0 # Initialize the overall best fitness to 0
        solution = None # Initialize the best solution (cluster assignments)
        population_size = len(components['ms_i'])  #Number of particles equals number of microservices/components

        # Run the PSO iteration multiple times (e.g., 60 times) to explore different initializations
        # and increase the chance of finding a good global optimum.
        for _ in range(60):
            # Run a single PSO optimization loop
            position, fitness = self.iteration(clusters, components, population_size)
            
            # If the current run yielded a feasible solution and its fitness is better
            # than the overall best fitness found so far, update the best solution.
            if fitness >= self.best_fitness:
                self.best_fitness = fitness
                solution = position

        return solution, self.best_fitness