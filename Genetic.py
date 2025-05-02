import Problem
import random
import numpy as np

class GeneticAlgorithm:

    def __init__(self, problem, population_size=100, num_generations=100, mutation_rate=0.01):
        self.problem = problem
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.problem.random_individual()
            population.append(individual)
        return population
    
    """
    def evaluate_population(self, population):
        fitness_indiv = []
        for individual in population:
            fitness_value = self.problem.evaluate(individual)
            if fitness_value is not None: 
                fitness_indiv.append((individual, fitness_value))
            else:
                fitness_indiv.append((individual, 0.0))
        return fitness_indiv
    """

    def evaluate_individual(self, individual):
        # Evaluate the individual using the problem's evaluate method
        fitness_value = self.problem.evaluate(individual)
        if fitness_value is not None: 
            return fitness_value
        else:
            return 0.0

    def evaluate_population(self, population):
        """Evaluates the entire population and returns list of (individual, fitness) tuples"""
        fitness_indiv = []
        for individual in population:
            fitness_value = self.evaluate_individual(individual)
            fitness_indiv.append((individual, fitness_value))
        return fitness_indiv

    # SELECTION ALGORITHMS
    
    def roulette_wheel_selection(self, evaluated_population, num_parents):
        """
        Roulette wheel selection (fitness proportionate selection)
        Higher fitness individuals have higher probability of selection
        
        Args:
            evaluated_population: List of (individual, fitness) tuples
            num_parents: Number of parents to select
            
        Returns:
            List of selected individuals for the mating pool
        """
        # Extract fitness values
        population, fitness_values = zip(*evaluated_population)
        population = list(population)
        
        # Handle negative fitness values if they exist by shifting all values
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            adjusted_fitness = [f - min_fitness + 1e-5 for f in fitness_values]
        else:
            adjusted_fitness = fitness_values
            
        # Calculate selection probabilities
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            # If all fitness values are 0, select randomly
            return random.choices(population, k=num_parents)
            
        selection_probs = [f/total_fitness for f in adjusted_fitness]
        
        # Select parents based on probabilities
        mating_pool = random.choices(population, weights=selection_probs, k=num_parents)
        return mating_pool
    
    def tournament_selection(self, evaluated_population, num_parents, tournament_size=3):
        """
        Tournament selection
        Randomly select tournament_size individuals and pick the best one
        Repeat until we have num_parents individuals
        
        Args:
            evaluated_population: List of (individual, fitness) tuples
            num_parents: Number of parents to select
            tournament_size: Number of individuals in each tournament
            
        Returns:
            List of selected individuals for the mating pool
        """
        mating_pool = []
        
        for _ in range(num_parents):
            # Randomly select tournament_size individuals
            tournament = random.sample(evaluated_population, min(tournament_size, len(evaluated_population)))
            
            # Find the winner (individual with highest fitness)
            winner = max(tournament, key=lambda x: x[1])[0]
            mating_pool.append(winner)
            
        return mating_pool
    
    def rank_selection(self, evaluated_population, num_parents):
        """
        Rank selection
        Selection probability is based on rank rather than absolute fitness
        Helps maintain diversity when fitness differences are extreme
        
        Args:
            evaluated_population: List of (individual, fitness) tuples
            num_parents: Number of parents to select
            
        Returns:
            List of selected individuals for the mating pool
        """
        # Sort population by fitness (ascending)
        sorted_population = sorted(evaluated_population, key=lambda x: x[1])
        
        # Assign ranks (higher rank = higher fitness)
        population_size = len(sorted_population)
        ranks = list(range(1, population_size + 1))
        
        # Calculate selection probabilities based on rank
        total_rank = sum(ranks)
        selection_probs = [r/total_rank for r in ranks]
        
        # Extract individuals from sorted population
        individuals = [individual for individual, _ in sorted_population]
        
        # Select parents based on rank probabilities
        mating_pool = random.choices(individuals, weights=selection_probs, k=num_parents)
        return mating_pool
    
    # PAIRING LOGIC
    
    def random_pairing(self, mating_pool):
        """
            random pairing method for the selected parents
            return list of pairs
        """
        # Make a copy of the mating pool to shuffle
        shuffled_pool = mating_pool.copy()
        random.shuffle(shuffled_pool)
        
        # Create pairs
        pairs = []
        for i in range(0, len(shuffled_pool) - 1, 2):
            pairs.append((shuffled_pool[i], shuffled_pool[i+1]))
            
        # If odd number of individuals, pair the last one with a random individual
        if len(shuffled_pool) % 2 != 0:
            pairs.append((shuffled_pool[-1], random.choice(shuffled_pool[:-1])))
            
        return pairs
    
    def fitness_based_pairing(self, evaluated_population, mating_pool):
        """
        Pairs individuals based on fitness similarity
        
        return: List of parent pairs (tuples)
        """
        # Create a fitness dictionary for quick lookup
        fitness_dict = {individual: fitness for individual, fitness in evaluated_population}
        
        # Sort mating pool by fitness
        sorted_pool = sorted(mating_pool, key=lambda ind: fitness_dict[ind])
        
        # Create pairs of adjacent individuals
        pairs = []
        for i in range(0, len(sorted_pool) - 1, 2):
            pairs.append((sorted_pool[i], sorted_pool[i+1]))
            
        # If odd number of individuals, pair the last one with a random individual
        if len(sorted_pool) % 2 != 0:
            pairs.append((sorted_pool[-1], random.choice(sorted_pool[:-1])))
            
        return pairs
    
    def diverse_pairing(self, evaluated_population, mating_pool):
        """
        Pairs individuals to maximize diversity
        Individuals with different fitness are paired together
        """
        
        fitness_dict = {individual: fitness for individual, fitness in evaluated_population}
        
        # Sort mating pool by fitness
        sorted_pool = sorted(mating_pool, key=lambda ind: fitness_dict[ind])
        
        
        pairs = []
        left, right = 0, len(sorted_pool) - 1
        
        while left < right:
            pairs.append((sorted_pool[left], sorted_pool[right]))
            left += 1
            right -= 1
            
        # If odd number of individuals, pair the middle one with a random individual
        if len(sorted_pool) % 2 != 0:
            middle = len(sorted_pool) // 2
            pairs.append((sorted_pool[middle], random.choice(sorted_pool[:middle] + sorted_pool[middle+1:])))
            
        return pairs
    
    # CROSSOVER
    
    def two_point_crossover(self, parent1, parent2, crossover_rate=0.8):
        """
        Two-point crossover operation
        Selects two random points and exchanges the segments between parents
        
        Args:
            parent1, parent2: The parent individuals
            crossover_rate: Probability of performing crossover
            
        Returns:
            Two offspring individuals
        """
        # Check if crossover should be performed
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Get parent length and ensure it's the same for both parents
        length = len(parent1)
        if length != len(parent2):
            raise ValueError("Parents must be of the same length for crossover")
        
        # Select two random crossover points
        if length < 3:
            # handle short chromosome
            point1 = random.randint(1, length - 1)
            point2 = point1
        else:
            points = sorted(random.sample(range(1, length), 2))
            point1, point2 = points
        
      
        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return offspring1, offspring2
    
    def run(self):
        """
        Runs the genetic algorithm
        
        Returns:
            The best individual found and its fitness
        """
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.num_generations):
            # Evaluate population
            evaluated_pop = self.evaluate_population(population)
            
            # Find and store the best individual
            best_individual, best_fitness = max(evaluated_pop, key=lambda x: x[1])
            
            print(f"Generation {generation}: Best fitness = {best_fitness}")
            
            # Create mating pool using selection
            mating_pool = self.tournament_selection(evaluated_pop, self.population_size)
            
            # Create parent pairs
            parent_pairs = self.random_pairing(mating_pool)
            
            # Create new population through crossover and mutation
            new_population = []
            
            for parent1, parent2 in parent_pairs:
                # Crossover
                offspring1, offspring2 = self.two_point_crossover(parent1, parent2)
                
                # Mutation (to be implemented)
                # offspring1 = self.mutate(offspring1)
                # offspring2 = self.mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # Limit new population to original size
            population = new_population[:self.population_size]
        
        # Evaluate final population
        final_evaluated_pop = self.evaluate_population(population)
        best_individual, best_fitness = max(final_evaluated_pop, key=lambda x: x[1])
        
        return best_individual, best_fitness
    
    """
    import numpy as np

elements = ['A', 'B', 'C']
probabilities = [0.5, 0.3, 0.2]  # Probabilities for A, B, and C respectively

# Select 5 elements with replacement
sample_with_replacement = np.random.choice(elements, size=5, replace=True, p=probabilities)
print("With replacement:", sample_with_replacement)

# Select 3 elements without replacement
sample_without_replacement = np.random.choice(elements, size=3, replace=False, p=probabilities)
print("Without replacement:", sample_without_replacement)

    """


    def crossover(self, fitness_indiv):
        parents = self.select_parents(fitness_indiv)
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                break
                
            parent1, parent2 = parents[i], parents[i+1]
            # Ensure the two parents are different
            if parent1 != parent2:
                offspring.append(self.crossover_parents(parent1, parent2))
            else:
                # If we have identical parents, try to find a different parent
                different_parent = None
                for j in range(len(parents)):
                    if j != i and j != i+1 and parents[j] != parent1:
                        different_parent = parents[j]
                        break
                
                if different_parent:
                    offspring.append(self.crossover_parents(parent1, different_parent))
                else:
                    # If all parents are identical, still perform crossover
                    # The crossover_parents method should handle tuple elements properly
                    offspring.append(self.crossover_parents(parent1, parent2))
        
        return offspring
    