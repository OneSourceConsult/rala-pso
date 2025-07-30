import numpy as np


class Particle:
    """
    Class representing a particle in the Particle Swarm Optimization (PSO) algorithm.
    """

    def __init__(self):
        """
        Initializes the particle with default values for position, fitness, velocity, best position, and best fitness.

        Attributes:
            position (int): The current position of the particle, representing a cluster.
            fitness (int): The fitness value of the particle.
            velocity (int): The velocity of the particle, used for position updates.
            best_position (int): The best position achieved by the particle so far.
            best_fitness (int): The best fitness value achieved by the particle so far.        
        """
        self.position = 0  # Particles current position (e.g., representing a cluster assignment)
        self.fitness = 0  # Particle's current fitness value
        self.velocity = 0  # Particle's velocity (determines how much the position changes)
        self.best_position = 0  # The best position found by this particle so far
        self.best_fitness = 0  # The best fitness value achieved by this particle so far

    def update_velocity(self, global_optimal: float, w: float, c1: float, c2: float):
        """
        Update the velocity of the particle based on its current position, best position, and global best position.

        Args:
            global_optimal (float): The optimal position found by the entire swarm (global best).
            w (float): The inertia weight, controlling the influence of the previous velocity.
            c1 (float): The cognitive learning factor (C1), weighting the pull towards the particle's own best position.
            c2 (float): The social learning factor (C2), weighting the pull towards the global best position.
        """
        # Calculate the momentum component (inertia of the previous velocity)
        monumentum = w * self.velocity
        # Calculate the cognitive component (pull towards the particle's own best position)
        memory = c1 * np.random.random() * (self.best_position - self.position)
        # Calculate the social component (pull towards the global best position)
        social = c2 * np.random.random() * (global_optimal - self.position)

        # Update the velocity using the sum of the three components
        updated_velocity = monumentum + memory + social

        self.velocity = updated_velocity

    def update_position(self, min_pos: int, max_pos: int):
        """
        Update the position of the particle based on its velocity.
        If the new position exceeds the maximum or minimum bounds, wrap around to the other side.

        Args:
            min_pos (int): The minimum allowed position for the particle.
            max_pos (int): The maximum allowed position for the particle.
        """

        # Update the position by adding the velocity and rounding to the nearest integer
        self.position = int(np.rint((self.position + self.velocity)))

        # Handle boundary conditions: if position goes out of bounds, wrap it around
        if self.position < min_pos:
            self.position = max_pos
        elif self.position > max_pos:
            self.position = min_pos

    def update_fitness(self, fitness: float):
        """
        Update the fitness of the particle. If the new fitness is better than the best fitness, update the best position and best fitness.

        Args:
            fitness (float): The newly calculated fitness value of the particle.
        """

        # If the new fitness is better than the current best fitness, update personal bests
        if fitness > self.best_fitness:
            self.update_best_fitness(fitness)  # Update the best fitness
            self.update_best_position()  # Update the best position to the current position
        self.fitness = fitness  # Always update the current fitness

    def update_best_position(self):
        """
        Update the best position of the particle to the current position.
        This is called when a new best fitness is found.
        """
        self.best_position = self.position

    def update_best_fitness(self, best_fitness: float):
        """
        Update the best fitness of the particle.

        Args:
            best_fitness (float): The new best fitness value to set for the particle.
        """
        self.best_fitness = best_fitness

    def get_position(self):
        """
        Get the current position of the particle.

        Returns:
            int: The current position of the particle.
        """
        return self.position

    def get_fitness(self):
        """
        Get the current fitness of the particle.

        Returns:
            float: The current fitness of the particle.
        """
        return self.fitness

    def get_best_position(self):
        """
        Get the best position of the particle.

        Returns:
            int: The best position achieved by the particle so far.

        """
        return self.best_position

    def get_best_fitness(self):
        """
        Get the best fitness of the particle.

        Returns:
            float: The best fitness value achieved by the particle so far.
        """
        return self.best_fitness

    def set_best_fitness(self, best_fitness: float):
        """
        Set the best fitness of the particle to a new value.
        This method allows directly setting the best fitness, which might be useful for initialization or specific scenarios.

        Args:
            best_fitness (float): The new best fitness value to set for the particle.
        """
        self.best_fitness = best_fitness

    def set_position(self, new_position: int):
        """
        Set the position of the particle to a new value.

        Args:
            new_position (int): The new position to set for the particle.
        """
        self.position = new_position

    def set_best_position(self, new_best_position: int):
        """
        Set the best position of the particle to a new value.

        Args:
            new_best_position (int): The new best position to set for the particle.
        """
        self.best_position = new_best_position

    def set_velocity(self, new_velocity: float):
        """
        Set the velocity of the particle to a new value.

        Args:
            new_velocity (float): The new velocity to set for the particle.
        """
        self.velocity = new_velocity
