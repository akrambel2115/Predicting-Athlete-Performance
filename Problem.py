import Node
import random

class AthletePerformanceProblem:
    """
    Defines a search problem for athlete performance planning.
    State: (day, fatigue, risk, performance)
    Actions: Train: (intensity_value, duration) or Rest: (0,0)
    Transition: deterministic model with tunable coefficients
    Cost: weighted sum of performance deficit, risk, and fatigue
    """
    def __init__(self,
                 initial_state: tuple = (0, 0.0, 0.0, 100.0),
                 alpha: float = 0.8,
                 beta: float = 0.001,
                 gamma: float = 0.1,
                 delta: float = 0.05,
                 epsilon: float = 0.001,
                 eta: float = 0.002,
                 theta: float = 0.1,
                 w1: float = 1.0,
                 w2: float = 1.0,
                 w3: float = 1.0,
                 w4: float = 1.0,
                 target_day: int = 30,
                 target_perf: float = 95.0,
                 max_fatigue: float = 0.5,
                 max_risk: float = 0.3):
        self.coeffs = dict(alpha=alpha, beta=beta, gamma=gamma,
                           delta=delta, epsilon=epsilon,
                           eta=eta, theta=theta)
        self.weights = dict(w1=w1, w2=w2, w3=w3)
        self.target_day = target_day
        self.target_perf = target_perf
        self.max_fatigue = max_fatigue
        self.max_risk = max_risk
        self.initial_state = initial_state

    def actions(self):
        train_actions = [(intensity, duration)
                        for intensity in (0.3, 0.6, 0.9)
                        for duration in (30, 60, 90)]
        return [train_actions, (0, 0)]  # Rest action

    def apply_action(self, state, action):
        day, fatigue, risk, performance = state
        intensity, duration = action
        coeffs = self.coeffs

        # Calculate new fatigue (F')
        new_fatigue = coeffs['alpha'] * fatigue + coeffs['beta'] * (intensity * duration) - coeffs['gamma'] * (1 - intensity)

        # Calculate new risk (R')
        new_risk = risk + coeffs['delta'] * new_fatigue + coeffs['epsilon'] * (intensity * duration)

        # Calculate new performance (P')
        new_performance = performance + coeffs['eta'] * (intensity * duration) - coeffs['theta'] * new_fatigue

        # Return the updated state
        return (day + 1, new_fatigue, new_risk, new_performance)
    
    def cost(self, state, action):
        w = self.weights
        _, fatigue, risk, performance = state
        _, fatigue_p, risk_p, performance_p = self.apply_action(state, action)
        delta_f, delta_r, delta_p = fatigue_p - fatigue, risk_p - risk, performance_p - performance

        # Calculate the cost as a weighted sum of performance deficit, risk, and fatigue
        return (w['w1'] * delta_f + w['w2'] * delta_r - w['w3'] * delta_p + w['w4'] * action[0] * action[1])
    
    def expand_node(self, node,use_cost=False, use_heuristic=False):
        """
        Expands a node by applying all possible actions and returning the resulting nodes.
        """
        children = []
        for action in self.actions():
            new_state = self.apply_action(node.state, action)
            if self.is_valid(new_state):
                cost = self.cost(node.state, action) if use_cost else 0
                heuristic = self.heuristic(new_state) if use_heuristic else 0
                child_node = Node(new_state, parent=node, action=action, cost=cost, f=cost + heuristic)
                children.append(child_node)
        return children

    def is_valid(self, state):
        day, fatigue, risk, performance = state
        return (
                0 <= fatigue <= 1.0
                and 0 <= risk <= 1.0
                and 0 <= performance <= 100.0)

    def is_goal(self, state):
        return (state.day == self.target_day
                and state.performance >= self.target_perf
                and state.fatigue <= self.max_fatigue
                and state.risk <= self.max_risk)
    
    
    def heuristic(self, state) -> float:
        max_I, max_D = 0.9, 90
        eta = self.coeffs['eta']
        remaining = max(0.0, self.target_perf - state.performance)
        best_gain_per_day = eta * max_I * max_D
        return remaining / best_gain_per_day if best_gain_per_day > 0 else 0.0

    def random_individual(self):

        schedule = []
        for _ in range(self.target_day):
            intensity = random.choice([0.3, 0.6, 0.9]) 
            duration = random.choice([30, 60, 90])
            if random.random() < 0.2: # 20% proba of rest day
                intensity = 0
                duration = 0
            schedule.append((intensity, duration))
        return tuple(schedule)
    
    def evaluate(self, individual):
        pass
    
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
    
    def evaluate_population(self, population):
        fitness_indiv = []
        for individual in population:
            fitness_value = self.problem.evaluate(individual)
            if fitness_value is not None: 
                fitness_indiv.append((individual, fitness_value))
            else:
                fitness_indiv.append((individual, 0.0))
        return fitness_indiv
    
    
    def select_parents(self, fitness_indiv):
        if not fitness_indiv:
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
    