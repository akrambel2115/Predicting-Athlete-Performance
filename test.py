fitness_indiv = [
    (((0.3, 90), (0.6, 60), (0.9, 90)), 144),
    (((0.3, 120), (0.3, 90), (0.9, 30)), 625),
    (((0.6, 120), (0.6, 60), (0.6, 60)), 25),
    (((0.3, 120), (0, 0), (0.9, 30)), 180),
                 ]

def select_parents(fitness_indiv):
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

print(select_parents(fitness_indiv))