import queue

class Node:
    def __init__(self, state, parent=None, action=None, cost=0, f=0, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.f = f
        self.g = g
        self.h = h

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state
    
    def __gt__(self, other):
        return isinstance(other, Node) and self.f > other.f

class AthletePerformanceProblem:
    """
    Defines a search problem for athlete performance planning.
    State: (day, fatigue, risk, performance)
    Actions: Train: (intensity_value, duration) or Rest: (0,0)
    Transition: deterministic model with tunable coefficients
    Cost: weighted sum of performance deficit, risk, and fatigue
    """
    def __init__(self,
                 initial_state: tuple = (0, 0.0, 0.0, 1.0),
                 target_day: int = 30,
                 target_perf: float = 95.0,
                 max_fatigue: float = 0.5,
                 max_risk: float = 0.3):
        
        self.target_day = target_day
        self.target_perf = target_perf
        self.max_fatigue = max_fatigue
        self.max_risk = max_risk
        self.initial_state = initial_state

    def actions(self):
        # (intensity, duration)
        return [(0, 0),
                (1, 30), (1, 60), (1, 90),
                (2, 30), (2, 60), (2, 90),
                (3, 30), (3, 60), (3, 90)]
        
    #TODO this was done by AI for testing, replace with yacine's transition model
    def transition_model(self, state, action):
        day, fatigue, risk, performance = state
        intensity, duration = action
        
        # Extract current state values
        new_day = day + 1  # Day always advances by 1
        
        # Basic fatigue model:
        # - Increases with training intensity and duration
        # - Decreases with rest (intensity 0)
        # - Limited recovery rate
        if intensity == 0:  # Rest day
            new_fatigue = max(0.0, fatigue - 0.1)  # Recovery
        else:
            fatigue_increase = intensity * duration * 0.001  # Scale with intensity and duration
            new_fatigue = min(1.0, fatigue + fatigue_increase)
        
        # Basic risk model:
        # - Increases with higher fatigue
        # - Higher intensity increases risk
        # - Limited recovery rate
        if intensity == 0:  # Rest day
            new_risk = max(0.0, risk - 0.05)  # Risk decreases with rest
        else:
            risk_factor = intensity * duration * 0.0005  # Base risk from training
            fatigue_multiplier = 1.0 + new_fatigue  # Fatigue amplifies risk
            new_risk = min(1.0, risk + (risk_factor * fatigue_multiplier))
        
        # Basic performance model:
        # - Training improves performance (with diminishing returns)
        # - High fatigue reduces performance gains
        # - Performance decays slightly without training
        if intensity == 0:  # Rest day
            new_performance = max(0.0, performance - 0.01)  # Slight decay
        else:
            # Training effect with diminishing returns
            training_effect = (intensity * duration * 0.0002) / (performance * 0.5)
            # Fatigue reduces training effect
            fatigue_penalty = new_fatigue * 0.5
            performance_change = training_effect * (1.0 - fatigue_penalty)
            new_performance = min(1.0, performance + performance_change)
        
        return (new_day, new_fatigue, new_risk, new_performance)

    def expand_node(self, node, use_cost=False, use_heuristic=False):
        children = []
        for action in self.actions():
            new_state = self.transition_model(node.state, action)
            if self.is_valid(new_state):
                cost = self.cost(new_state) if use_cost else 0
                heuristic = self.heuristic(new_state) if use_heuristic else 0
                f = cost + heuristic
                
                new_node = Node(
                    state=new_state,
                    parent=node,
                    action=action,
                    cost=cost,
                    f=f,
                    g=cost,
                    h=heuristic
                )
                
                children.append(new_node)
                
        return children

    def is_valid(self, state):
        _, fatigue, risk, _ = state
        return fatigue <= self.max_fatigue and risk <= self.max_risk

    def is_goal(self, state):
        day, _, _, performance = state
        return day >= self.target_day and performance >= self.target_perf
    
    def cost(self, state):
        """
        Cost function based on the formula: c(s, a, s') = w1(1-P') + w2R' + w3F'.
        where:
        - P' is performance (higher is better)
        - R' is injury risk (lower is better)
        - F' is fatigue level (lower is better)
        - w1, w2, w3 are weights for each factor
        
        This penalizes lower performance, higher risk, higher fatigue.
        """
        day, fatigue, risk, performance = state
        
        # TODO: tune weights
        w1, w2, w3 = 0.4, 0.3, 0.3
        
        days_remaining = max(0, self.target_day - day)
        urgency_factor = 1.0
        if days_remaining > 0:
            urgency_factor = 1.0 + (1.0 / days_remaining) if days_remaining < 10 else 1.0
        
        return urgency_factor * (w1 * (1 - performance) + w2 * risk + w3 * fatigue)
    
    def heuristic(self, state) -> float:
        """
        Heuristic function that estimates remaining cost to reach the goal
        Based on the same formula as cost: w1(1-P') + w2R' + w3F'
        
        This estimates how far the athlete is from ideal performance while
        accounting for current risk and fatigue levels and remaining days.
        """
        day, fatigue, risk, performance = state

        target = self.target_perf
        
        perf_gap = max(0, target - performance) 
        
        # TODO should be same weights used in cost to ensure admissibility and consistency
        w1, w2, w3 = 0.4, 0.3, 0.3
        
        # the closer to target day, the more urgent it is to reach performance
        days_remaining = max(0, self.target_day - day)
        urgency_factor = 1.0
        if days_remaining > 0:
            urgency_factor = 1.0 + (1.0 / days_remaining) if days_remaining < 10 else 1.0
            
        return urgency_factor * (w1 * perf_gap + w2 * (self.max_risk - risk) + w3 * (self.max_fatigue - fatigue))

class GreedySearch:
    def __init__(self, problem):
        """
        Initialize the greedy search process with a problem instance.

        Input Parameters:
            - problem: An instance of AthletePerformanceProblem.
        """
        self.problem = problem

    def search(self, max_depth=float('inf')):
        """
        Execute a greedy best-first search based on the heuristic function.

        Input Parameters:
            - max_depth: An integer indicating the maximum depth to search.

        Output:
            - A Node instance representing the goal if found, otherwise best alternative.
        """
        frontier = queue.PriorityQueue()
        explored = set()
        initial_node = Node(self.problem.initial_state)
        
        # Set initial heuristic
        initial_node.h = self.problem.heuristic(initial_node.state)
        initial_node.f = initial_node.h
        
        # Add to frontier with priority based on heuristic
        frontier.put((initial_node.f, id(initial_node), initial_node))
        
        # Track states in frontier to avoid duplicates
        frontier_states = {self._round_state(initial_node.state)}  # MODIFIED: Rounded state tracking
        
        # Track best state (even if not goal)
        best_node = initial_node
        best_score = self.problem.cost(initial_node.state)  # MODIFIED: Use cost for best estimation

        iteration = 0  # MODIFIED: Track iterations

        while not frontier.empty():
            iteration += 1
            if iteration % 100 == 0:
                print(f"[Search iteration {iteration}] Frontier size: {frontier.qsize()}")

            _, _, current_node = frontier.get()
            current_state = current_node.state
            rounded_state = self._round_state(current_state)  # MODIFIED

            frontier_states.discard(rounded_state)
            explored.add(rounded_state)

            if self.problem.is_goal(current_state):
                print(f"Goal reached: {current_state}")
                return current_node

            # Update best node found so far
            score = self.problem.cost(current_state)
            if score < best_score:
                best_score = score
                best_node = current_node

            day = current_state[0]
            if current_node.depth > max_depth or day >= self.problem.target_day:
                continue

            # Expand children
            children = self.problem.expand_node(current_node, use_cost=False, use_heuristic=True)

            for child in children:
                child_state = child.state
                rounded_child_state = self._round_state(child_state)  # MODIFIED
                if rounded_child_state not in explored and rounded_child_state not in frontier_states:
                    frontier.put((child.f, id(child), child))
                    frontier_states.add(rounded_child_state)

        print("Search ended without exact goal. Returning best node found.")
        return best_node  # MODIFIED: Always return best node found

    # ADD THIS METHOD TO YOUR CLASS
    def _round_state(self, state):
        """
        Helper to reduce state space size by rounding float values.
        """
        day, fatigue, risk, performance = state
        return (day, round(fatigue, 2), round(risk, 2), round(performance, 2))  # MODIFIED


    def reconstruct_path(self, node):
        """
        Reconstruct the path from the initial state to the goal state.
        """
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return path[::-1]  # reverse to get the correct order

def test_greedy_search():
    """
    Test the greedy search algorithm.
    """
    
    print("Testing Greedy Search Algorithm")
    print("-----------------------------------------")
    
    problem = AthletePerformanceProblem(
        initial_state=(0, 0.0, 0.0, 0.4),  
        target_day=30,                     
        target_perf=0.9,                   
        max_fatigue=0.6,                   
        max_risk=0.3                      
    )
    
    searcher = GreedySearch(problem)
    
    print("Running Greedy Best-First Search...")
    goal_node = searcher.search()
    
    if goal_node is None:
        print("No solution found.")
    else:
        path = searcher.reconstruct_path(goal_node)
        
        print("\nTraining Plan:")
        print("Day | Intensity | Duration | Fatigue | Risk | Performance")
        print("----|-----------|----------|---------|------|------------")
        
        state = problem.initial_state
        day = 0
        print(f"{day:3d} | {'-':9} | {'-':8} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")
        
        for action in path:
            if action:  # Skip first None action
                state = problem.transition_model(state, action)
                day += 1
                intensity, duration = action
                print(f"{day:3d} | {intensity:9d} | {duration:8d} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")
        
        final_day, final_fatigue, final_risk, final_perf = state
        print("\nFinal State:")
        print(f"Day: {final_day}")
        print(f"Fatigue: {final_fatigue:.2f}")
        print(f"Risk: {final_risk:.2f}")
        print(f"Performance: {final_perf:.2f}")
        
        if problem.is_goal(state):
            print("\nGoal achieved!")
        else:
            print("\nGoal not achieved.")

# Run the test if this file is executed directly
if __name__ == "__main__":
    test_greedy_search()
#TODO change the logic for the alternative solution to return lst explored node instead
