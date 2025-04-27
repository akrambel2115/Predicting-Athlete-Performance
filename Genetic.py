import Problem
import random

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
    
    
    
    
    def select_parents(self, population):
        if not population:
            return []
            
        fitness_sum_avg = sum(fitness for _, fitness in fitness_indiv) / len(fitness_indiv)
        if fitness_sum_avg == 0:

            return [ind for ind, _ in fitness_indiv[:min(6, len(fitness_indiv))]]
            
        initial_count = [round(fitness / fitness_sum_avg) for _, fitness in fitness_indiv]
        
        # sort by fitness in descending order
        sorted_fitness = sorted([(ind, count, fitness) for (ind, fitness), count in zip(fitness_indiv, initial_count)], 
                            key=lambda x: x[2], reverse=True)
        
        individuals_counts = [(ind, min(count, 2)) for ind, count, _ in sorted_fitness]
        
        valid_individuals = [ind for ind, count in individuals_counts if count > 0]
        
        if len(valid_individuals) < 3:
            for i in range(min(3, len(individuals_counts))):
                if individuals_counts[i][1] == 0:
                    individuals_counts[i] = (individuals_counts[i][0], 1)
        
        parents = []
        for individual, count in individuals_counts:
            if count > 0:
                parents.extend([individual] * count)
        
        # even number of parents
        if len(parents) % 2 != 0 and len(parents) > 0:
            parents.pop()
            
        return parents


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
    