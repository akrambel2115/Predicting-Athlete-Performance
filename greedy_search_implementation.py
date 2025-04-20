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
        _, fatigue, risk, performance = state
        
        # TODO: tune weights
        w1, w2, w3 = 0.4, 0.3, 0.3
        
        return w1 * (1 - performance) + w2 * risk + w3 * fatigue
    
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
            
        return urgency_factor * (w1 * perf_gap + w2 * risk + w3 * fatigue)

class GeneralSearch:
    def __init__(self, problem):
        """
        Initialize the general search process with a problem instance.

        Input Parameters:
            - problem: An instance of AthletePerformanceProblem.

        Output:
            - An instance of GeneralSearch with default flags for cost and heuristic usage.
        """
        self.problem = problem
        self.use_cost = True
        self.use_heuristic = False

    def set_frontier(self, search_strategy="best_first"):
        """
        Set up the frontier based on the chosen search strategy (only supports  greedy best-first for now).

        Input Parameters:
            - search_strategy: A string specifying the strategy.

        Output:
            - A frontier appropriate for the strategy.
        """

        if search_strategy == "best_first":
            frontier = queue.PriorityQueue()
            self.use_cost = False
            self.use_heuristic = True

        else:
            raise ValueError("Unsupported search strategy: " + str(search_strategy))

        return frontier

    def search(self, search_strategy="best-first", max_depth=float('inf')):
        """
        Execute a general graph search based on the specified strategy (only supports greedy search for now).

        Input Parameters:
            - search_strategy: A string indicating which strategy to use.
            - max_depth: An integer indicating the maximum depth to search.

        Output:
            - A Node instance representing the goal if found, otherwise None.
        """
        frontier = self.set_frontier(search_strategy)
        explored = set()
        initial_node = Node(self.problem.initial_state)
        
        # Set initial cost and heuristic if needed
        if self.use_cost:
            initial_node.g = self.problem.cost(initial_node.state)
        if self.use_heuristic:
            initial_node.h = self.problem.heuristic(initial_node.state)
            
        initial_node.f = initial_node.g + initial_node.h
        
        # Add to frontier with appropriate priority if needed
        frontier.put((initial_node.f, id(initial_node), initial_node))
        
        # Track states in frontier to avoid duplicates
        frontier_states = {initial_node.state}
        
        # Track best solution at target day
        target_day_nodes = []
        best_node = initial_node
        best_score = float('inf')
        
        while not frontier.empty():
            # Get next node from frontier
            current = frontier.get()
            _, _, current_node = current
                
            current_state = current_node.state
            if current_state in frontier_states:
                frontier_states.remove(current_state)
            
            # Check if exceeding max depth
            if current_node.depth > max_depth:
                continue
            
            # Extract day from state
            day = current_state[0]
            
            # For AthletePerformanceProblem, check if exceeding target day
            if day > self.problem.target_day:
                continue
            
            # Add to explored
            explored.add(current_state)
            
            # Check if goal reached
            if self.problem.is_goal(current_state):
                print(f"Goal reached: {current_state}")
                return current_node
            
            # For nodes at target day, keep track of them
            if day == self.problem.target_day:
                target_day_nodes.append(current_node)
                
                # Calculate score (higher performance is better, lower risk and fatigue are better)
                _, fatigue, risk, performance = current_state
                score = (-0.6 * performance) + (0.2 * risk) + (0.2 * fatigue)
                
                if score < best_score:
                    best_score = score
                    best_node = current_node
            
            # Only expand nodes that haven't reached the target day yet
            if day < self.problem.target_day:
                # Expand node and add children to frontier
                children = self.problem.expand_node(current_node, self.use_cost, self.use_heuristic)
                
                for child in children:
                    child_state = child.state
                    
                    # Only consider states that:
                    # 1. Don't exceed target day
                    # 2. Haven't been explored
                    # 3. Aren't already in frontier
                    
                    child_day = child_state[0]
                    if (child_day <= self.problem.target_day and 
                        child_state not in explored and
                        child_state not in frontier_states):
                        
                        if search_strategy in ["uniform_cost", "best_first", "A*"]:
                            frontier.put((child.f, id(child), child))
                        else:
                            frontier.put(child)
                        frontier_states.add(child_state)
        
        # When search completes, analyze target day nodes if we have any
        if target_day_nodes:
            # Check if any node at target day is a goal
            goal_nodes = [node for node in target_day_nodes if self.problem.is_goal(node.state)]
            if goal_nodes:
                # Find the best goal node based on score
                best_goal_node = goal_nodes[0]
                best_goal_score = float('inf')
                
                for node in goal_nodes:
                    _, fatigue, risk, performance = node.state
                    score = (-0.6 * performance) + (0.2 * risk) + (0.2 * fatigue)
                    if score < best_goal_score:
                        best_goal_score = score
                        best_goal_node = node
                
                print(f"Goal reached at target day: {best_goal_node.state}")
                return best_goal_node
            
            # If no goal node, return the best node at target day
            print(f"No goal reached, returning best alternative at target day: ")
            return best_node
        
        print("Search terminated with no solution found.")
        return None

    def _reconstruct_path(self, node):
        """
        Reconstruct the path from the initial state to the goal state.
        """
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return path[::-1]  # reverse to get the correct order

def test_general_search():
    """
    Test the general search algorithm with different strategies.
    """
    
    print("Testing General Search Algorithm")
    print("-----------------------------------------")
    
    problem = AthletePerformanceProblem(
        initial_state=(0, 0.0, 0.0, 0.4),  
        target_day=10,                     
        target_perf=0.9,                   
        max_fatigue=0.6,                   
        max_risk=0.3                      
    )
    
    searcher = GeneralSearch(problem)
    
    print("Running Greedy Best-First Search...")
    goal_node = searcher.search(search_strategy="best_first")
    
    if goal_node is None:
        print("No solution found.")
    else:
        path = searcher._reconstruct_path(goal_node)
        
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
    test_general_search()
