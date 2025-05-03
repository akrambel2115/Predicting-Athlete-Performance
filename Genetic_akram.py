import Node
import Problem
import random
import time

class GeneticAlgorithm:

    def __init__(self, problem, population_size=100, num_generations=100, mutation_rate=0.01):
        self.problem = problem
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.state_cache = {}  # Cache for storing intermediate states
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.problem.random_individual()
            population.append(individual)
        return population
    
    def evaluate_individual(self, individual, cached_state=None, start_index=0):
        """
        Evaluate an individual and return a fitness tuple of (performance, risk, fatigue)
        Higher performance is better, lower risk and fatigue are better
        
        Args:
            individual: The training schedule to evaluate
            cached_state: Optional cached state to start from (to avoid recomputing)
            start_index: Index to start evaluation from (for partial evaluation)
            
        Returns:
            Tuple of (performance, risk, fatigue)
        """
        if not individual:
            return (0.0, 1.0, 5.0)  # Worst case values
        
        # Make a copy to avoid modifying the original individual
        individual_copy = list(individual.copy())
        
        # Check if we've already evaluated this exact schedule
        individual_key = tuple(individual_copy)
        if individual_key in self.state_cache:
            return self.state_cache[individual_key]
        
        # If we have a cached state to start from
        if cached_state and start_index > 0:
            # Use cached state directly
            state = cached_state
        else:
            # Start from initial state
            state = self.problem.initial_state
        
        # Apply each training action in sequence
        for i, action in enumerate(individual_copy[start_index:], start=start_index):
            if i >= self.problem.target_day:
                break
                
            # Apply the action and get the next state - history is now part of state
            next_state = self.problem.apply_action(state, action)
            state = next_state
            
            # Early termination if invalid state (using first 4 elements)
            if not self.problem.is_valid(state):
                day, fatigue, risk, performance, _ = state
                return (max(0.1, performance * 0.5), min(1.0, risk * 1.2), min(5.0, fatigue * 1.2))
        
        # Extract final state values (day, fatigue, risk, performance, _)
        day, fatigue, risk, performance, _ = state
        
        # Cache the final result
        fitness = (performance, risk, fatigue)
        self.state_cache[individual_key] = fitness
        
        return fitness
    
    def find_closest_cached_state(self, individual):
        """
        Find the closest cached state for a given individual to optimize evaluation
        """
        # Look for cached prefix states
        best_match = None
        best_match_length = 0
        
        for i in range(len(individual), 0, -5):  # Check every 5 days for a match
            prefix = tuple(individual[:i])
            key = (prefix, i)
            if key in self.state_cache:
                return self.state_cache[key], i
        
        return None, 0
        
    def fitness_comparison(self, fitness_tuple1, fitness_tuple2):
        """
        Compare two fitness tuples and return True if the first is better
        Considers performance (higher is better), risk (lower is better), and fatigue (lower is better)
        """
        perf1, risk1, fatigue1 = fitness_tuple1
        perf2, risk2, fatigue2 = fitness_tuple2
        
        # Check if one clearly dominates the other
        if perf1 > perf2 and risk1 <= risk2 and fatigue1 <= fatigue2:
            return True
        if perf2 > perf1 and risk2 <= risk1 and fatigue2 <= fatigue1:
            return False
        
        # Calculate weighted score (prioritize performance)
        # Negative weights for risk and fatigue (lower is better)
        perf_weight = 2.0
        risk_weight = -1.5  # Negative because lower is better
        fatigue_weight = -1.0  # Negative because lower is better
        
        score1 = perf_weight * perf1 + risk_weight * risk1 + fatigue_weight * fatigue1
        score2 = perf_weight * perf2 + risk_weight * risk2 + fatigue_weight * fatigue2
        
        return score1 > score2
    
    def select_parents(self, population):
        if not population:
            return []
        
        # Evaluate all individuals in the population
        fitness_indiv = [(ind, self.evaluate_individual(ind)) for ind in population]
        
        # Sort individuals by fitness (using custom comparison)
        sorted_individuals = sorted(
            fitness_indiv, 
            key=lambda x: x[1],  # Sort by fitness tuple
            reverse=True  # Higher performance first
        )
        
        # Use the top 50% of individuals as potential parents
        top_half = sorted_individuals[:len(sorted_individuals)//2]
        
        # Ensure we have at least some parents
        if not top_half:
            return [ind for ind, _ in sorted_individuals[:min(6, len(sorted_individuals))]]
        
        # Calculate selection probabilities based on rank
        total_ranks = sum(range(1, len(top_half) + 1))
        
        # Select parents with probability proportional to their rank
        parents = []
        for i, (ind, _) in enumerate(top_half):
            # Higher ranks get more copies (rank = position from bottom)
            copies = max(1, round(2 * (len(top_half) - i) / len(top_half)))
            parents.extend([ind] * min(copies, 3))  # Limit to max 3 copies
        
        # Ensure even number of parents for pairing
        if len(parents) % 2 != 0 and len(parents) > 0:
            parents.pop()
        
        return parents
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents by swapping a subset of days.
        This version treats each day's (intensity, duration) as an atomic unit.
        """
        # Convert parents to lists to ensure consistent types
        parent1 = list(parent1)
        parent2 = list(parent2)
        
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        if len(parent1) <= 1:
            return parent1, parent2
        
        # Select crossover point (not at the start or end)
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Create children by combining parent genetic material
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, individual):
        """
        Apply random mutations to an individual based on the mutation rate.
        This version treats each day's (intensity, duration) as an atomic unit.
        """
        # Make sure we're working with a list
        mutated = list(individual)
        
        for i in range(len(mutated)):
            # Apply mutation with probability = mutation_rate
            if random.random() < self.mutation_rate:
                # Mutate the entire (intensity, duration) pair as a single unit
                current_pair = mutated[i]
                intensity, duration = current_pair
                
                # Generate a new pair that's different from the current one
                if intensity == 0:  # If it's a rest day
                    # Change to a training day
                    new_intensity = random.choice([0.3, 0.6, 0.9])
                    new_duration = random.choice([30, 60, 90, 120])
                else:  # If it's a training day
                    # Either change intensity/duration or make it a rest day
                    if random.random() < 0.2:  # 20% chance to make it a rest day
                        new_intensity = 0.0
                        new_duration = 0
                    else:
                        # Change intensity or duration or both
                        intensities = [0.3, 0.6, 0.9]
                        if intensity in intensities:
                            intensities.remove(intensity)
                        durations = [30, 60, 90, 120]
                        if duration in durations:
                            durations.remove(duration)
                            
                        # Decide what to change
                        change_type = random.choice(["intensity", "duration", "both"])
                        
                        if change_type in ["intensity", "both"]:
                            new_intensity = random.choice(intensities)
                        else:
                            new_intensity = intensity
                            
                        if change_type in ["duration", "both"]:
                            new_duration = random.choice(durations)
                        else:
                            new_duration = duration
                
                mutated[i] = (new_intensity, new_duration)
        
        # Return as a list, not a tuple
        return mutated
    
    def evaluate_population(self, population):
        """
        Evaluate all individuals in the population efficiently with caching.
        """
        result = []
        for ind in population:
            # Try to find a cached state to start from
            cached_state, start_idx = self.find_closest_cached_state(ind)
            fitness = self.evaluate_individual(ind, cached_state, start_idx)
            result.append((ind, fitness))
        return result
    
    def get_best_individual(self, population):
        """
        Find the individual with the highest fitness in the population.
        """
        if not population:
            return None
        
        population_fitness = self.evaluate_population(population)
        
        # Sort by fitness using our custom comparison
        sorted_pop = sorted(population_fitness, key=lambda x: x[1], reverse=True)
        return sorted_pop[0][0]  # Return the best individual
    
    def run(self):
        """
        Run the genetic algorithm for a specified number of generations with performance optimizations.
        """
        # Initialize population
        population = self.initialize_population()
        best_individual = None
        best_fitness = (0.0, 1.0, 5.0)  # Worst case (perf, risk, fatigue)
        
        # Clear cache before starting
        self.state_cache = {}
        
        # Track progress over generations
        performance_history = []
        risk_history = []
        fatigue_history = []
        
        # Track progress
        for generation in range(self.num_generations):
            start_time = time.time()
            
            # Evaluate all individuals
            population_fitness = self.evaluate_population(population)
            
            # Sort by fitness
            sorted_population = sorted(population_fitness, key=lambda x: x[1], reverse=True)
            
            # Current best individual and fitness
            current_best_individual = sorted_population[0][0]
            current_best_fitness = sorted_population[0][1]
            
            # Record current generation's best values
            performance, risk, fatigue = current_best_fitness
            performance_history.append(performance)
            risk_history.append(risk)
            fatigue_history.append(fatigue)
            
            # Update overall best if better
            if best_individual is None or self.fitness_comparison(current_best_fitness, best_fitness):
                best_individual = current_best_individual
                best_fitness = current_best_fitness
                generation_found = generation
                print(f"New best solution found in generation {generation}!")
                print(f"Performance: {best_fitness[0]:.2f}, Risk: {best_fitness[1]:.2f}, Fatigue: {best_fitness[2]:.2f}")
            
            # Print progress every 10 generations with timing info
            if generation % 10 == 0:
                gen_time = time.time() - start_time
                print(f"Generation {generation}: Perf={performance:.2f}, Risk={risk:.2f}, Fat={fatigue:.2f}, Time={gen_time:.2f}s")
            
            # Don't stop early, run all generations
            if generation == self.num_generations - 1:
                break
                
            # Select parents based on fitness
            parents = self.select_parents(population)
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Elitism: keep the best individual and some top performers
            elites_count = max(1, self.population_size // 20)  # Keep top 5%
            new_population.extend([ind for ind, _ in sorted_population[:elites_count]])
            
            # Fill the rest of the population with offspring
            while len(new_population) < self.population_size:
                # Select parents for crossover
                if len(parents) < 2:
                    # If not enough parents, add random individuals
                    new_population.append(self.problem.random_individual())
                    continue
                
                # Select two parents
                idx1 = random.randint(0, len(parents) - 1)
                idx2 = random.randint(0, len(parents) - 1)
                while idx2 == idx1 and len(parents) > 1:
                    idx2 = random.randint(0, len(parents) - 1)
                
                parent1 = parents[idx1]
                parent2 = parents[idx2]
                
                # Perform crossover with 90% probability
                if random.random() < 0.9:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    # Sometimes skip crossover to maintain diversity
                    child1, child2 = parent1, parent2
                
                # Perform mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add children to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace old population with new population
            population = new_population
        
        # Print final statistics about the run
        print("\nGenetic Algorithm Run Complete")
        print(f"Best solution found in generation {generation_found} out of {self.num_generations}")
        print(f"Best Performance: {best_fitness[0]:.2f}")
        print(f"Best Risk: {best_fitness[1]:.2f}")
        print(f"Best Fatigue: {best_fitness[2]:.2f}")
        
        # Calculate improvement from first to last generation
        if performance_history:
            first_perf, last_perf = performance_history[0], performance_history[-1]
            first_risk, last_risk = risk_history[0], risk_history[-1]
            first_fat, last_fat = fatigue_history[0], fatigue_history[-1]
            
            print("\nImprovement Summary:")
            print(f"Performance: {first_perf:.2f} → {last_perf:.2f} ({(last_perf-first_perf):.2f} change)")
            print(f"Risk: {first_risk:.2f} → {last_risk:.2f} ({(last_risk-first_risk):.2f} change)")
            print(f"Fatigue: {first_fat:.2f} → {last_fat:.2f} ({(last_fat-first_fat):.2f} change)")
        
        # Return the best individual found
        return best_individual, best_fitness

def test_genetic_algorithm(days=14, pop_size=50, generations=30, mutation_rate=0.05):
    """
    Test the genetic algorithm on the athlete performance problem.
    
    Args:
        days: Number of days in the training plan
        pop_size: Size of the population
        generations: Number of generations to run
        mutation_rate: Probability of mutation
        
    Returns:
        The best schedule found and its fitness value
    """
    # Import here to avoid circular imports
    from Problem import AthletePerformanceProblem
    
    print(f"Testing Genetic Algorithm with:")
    print(f"- Days: {days}")
    print(f"- Population Size: {pop_size}")
    print(f"- Generations: {generations}")
    print(f"- Mutation Rate: {mutation_rate}")
    print("-" * 50)
    
    start_time = time.time()
    
    # Set up the problem with target day and constraints
    problem = AthletePerformanceProblem(
        initial_state=(0, 1.0, 0.1, 5.0),  # day, fatigue, risk, performance
        genetic=True
    )
    # Set the target parameters
    problem.target_day = days
    problem.target_perf = 9.0
    problem.max_fatigue = 3
    problem.max_risk = 0.2
    
    # Create and run the genetic algorithm
    ga = GeneticAlgorithm(
        problem=problem,
        population_size=pop_size,
        num_generations=generations,
        mutation_rate=mutation_rate
    )
    
    best_schedule, best_fitness = ga.run()
    performance, risk, fatigue = best_fitness
    
    total_time = time.time() - start_time
    
    # Print the results
    print("\nBest Schedule Found:")
    print("-" * 50)
    print("Day | Intensity | Duration | Description")
    print("-" * 50)
    
    for i, (intensity, duration) in enumerate(best_schedule):
        if intensity == 0:
            desc = "Rest Day"
        elif intensity <= 0.3:
            desc = "Light Training"
        elif intensity <= 0.6:
            desc = "Moderate Training"
        else:
            desc = "Intense Training"
        
        print(f"{i+1:3d} | {intensity:9.1f} | {duration:8.0f} | {desc}")
    
    print("-" * 50)
    print("\nFinal State:")
    print(f"Performance: {performance:.2f}")
    print(f"Risk: {risk:.2f}")
    print(f"Fatigue: {fatigue:.2f}")
    
    # Calculate a weighted fitness score for display
    weighted_score = (2.0 * performance) - (1.5 * risk) - (1.0 * fatigue)
    print(f"Weighted Fitness Score: {weighted_score:.2f}")
    
    # Check if goals were met
    print("\nGoal Analysis:")
    print(f"Performance Goal ({problem.target_perf:.1f}): {'✓' if performance >= problem.target_perf else '✗'}")
    print(f"Max Fatigue Goal ({problem.max_fatigue:.1f}): {'✓' if fatigue <= problem.max_fatigue else '✗'}")
    print(f"Max Risk Goal ({problem.max_risk:.2f}): {'✓' if risk <= problem.max_risk else '✗'}")
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Cache size: {len(ga.state_cache)} entries")
    
    return best_schedule, best_fitness

if __name__ == "__main__":
    # Run the test function when the script is executed directly
    test_genetic_algorithm()