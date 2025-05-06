import Node
import Problem
import heapq
import time
import functools
import threading

class AStarSearch:
    """
    A* search algorithm implementation with ML prediction caching for efficiency.
    Uses a priority queue to explore nodes in order of f-score (g + h).
    """
    def __init__(self, problem):
        self.problem = problem
        self.heuristic_cache = {}  # Cache for heuristic values
        self.expanded_nodes = 0
        self.max_queue_size = 0
        self.execution_time = 0
        
        # ML model prediction cache
        self.prediction_cache = {}
        self.prediction_lock = threading.Lock()  # Lock for thread safety
        self.prediction_hits = 0
        self.prediction_misses = 0
        
        # Set default target parameters if not already set
        if not hasattr(self.problem, 'target_day'):
            self.problem.target_day = 14
        if not hasattr(self.problem, 'target_perf'):
            self.problem.target_perf = 7.0
        if not hasattr(self.problem, 'max_fatigue'):
            self.problem.max_fatigue = 3.5
        if not hasattr(self.problem, 'max_risk'):
            self.problem.max_risk = 0.5
        
        # Apply monkey patch to problem's apply_action for caching
        self._patch_apply_action()
    
    def _patch_apply_action(self):
        """Monkey patch the problem's apply_action method to add caching"""
        original_apply_action = self.problem.apply_action
        
        @functools.wraps(original_apply_action)
        def cached_apply_action(state, action):
            # Create a cache key from relevant state components and action
            day, fatigue, risk, performance, history = state
            
            # Use only last state for caching to reduce key size
            if history and len(history) > 0:
                last_history_item = history[-1]
            else:
                last_history_item = {}
                
            # Create a hashable cache key
            cache_key = (day, round(fatigue, 2), round(risk, 2), round(performance, 2), 
                        tuple(sorted(last_history_item.items())) if last_history_item else None,
                        action)
            
            # Check if we already computed this state transition
            with self.prediction_lock:
                if cache_key in self.prediction_cache:
                    self.prediction_hits += 1
                    return self.prediction_cache[cache_key]
                
                # If not in cache, compute it
                self.prediction_misses += 1
                result = original_apply_action(state, action)
                
                # Cache the result if cache size is reasonable
                if len(self.prediction_cache) < 100000:  # Set a reasonable cache size limit
                    self.prediction_cache[cache_key] = result
                    
                return result
                
        # Replace the original method with our cached version
        self.problem.apply_action = cached_apply_action
    
    def get_state_key(self, state):
        """Creates a hashable key from a state by rounding numeric values."""
        day, fatigue, risk, performance, _ = state
        return (day, round(fatigue, 1), round(risk, 2), round(performance, 1))
    
    def get_heuristic(self, state):
        """Gets the heuristic value for a state, using cache when possible."""
        state_key = self.get_state_key(state)
        if state_key in self.heuristic_cache:
            return self.heuristic_cache[state_key]
        
        h_value = self.problem.heuristic(state)
        self.heuristic_cache[state_key] = h_value
        return h_value
    
    def search(self, max_iterations=10000, time_limit=240, exact_days=True):
        """
        Performs A* search to find an optimal training plan.
        
        Args:
            max_iterations: Maximum number of iterations to prevent infinite loops
            time_limit: Maximum time in seconds to run search
            exact_days: Whether to require exact day match for goal state
            
        Returns:
            The goal node if a solution is found, None otherwise
        """
        start_time = time.time()
        
        # Initialize the start node
        start_node = Node.Node(self.problem.initial_state)
        start_node.g = 0  # Cost from start to start is 0
        start_node.h = self.get_heuristic(start_node.state)
        start_node.f = start_node.g + start_node.h
        
        # Initialize open and closed sets
        open_set = []  # Priority queue
        heapq.heappush(open_set, (start_node.f, id(start_node), start_node))  # Using object ID as tiebreaker
        
        # Use a dictionary for O(1) lookups in the open set
        open_dict = {self.get_state_key(start_node.state): start_node}
        
        # Closed set to track visited states
        closed_set = set()
        
        iterations = 0
        while open_set and iterations < max_iterations:
            # Check time limit
            if time.time() - start_time > time_limit:
                print(f"Search time limit reached ({time_limit}s)")
                break
                
            iterations += 1
            
            # Track max queue size for metrics
            self.max_queue_size = max(self.max_queue_size, len(open_set))
            
            # Get node with lowest f-score
            _, _, current_node = heapq.heappop(open_set)
            current_key = self.get_state_key(current_node.state)
            
            # Remove from open dictionary
            if current_key in open_dict:
                del open_dict[current_key]
            
            # Check if we reached a goal state
            day, fatigue, risk, performance, _ = current_node.state
            
            # Modified goal state check to enforce exact day match if requested
            day_condition = (day == self.problem.target_day) if exact_days else (day >= self.problem.target_day)
            
            if (day_condition and 
                performance >= self.problem.target_perf and
                fatigue <= self.problem.max_fatigue and
                risk <= self.problem.max_risk):
                self.execution_time = time.time() - start_time
                print(f"ML prediction cache: {self.prediction_hits} hits, {self.prediction_misses} misses")
                return current_node
            
            # Skip if we've already processed this state
            if current_key in closed_set:
                continue
            
            # Add to closed set
            closed_set.add(current_key)
            self.expanded_nodes += 1
            
            # Print progress periodically
            if self.expanded_nodes % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Expanded {self.expanded_nodes} nodes, queue size: {len(open_set)}, time: {elapsed:.1f}s")
            
            # Get all possible actions for the current state
            for action in self.problem.actions():
                # Apply the action to get a new state
                new_state = self.problem.apply_action(current_node.state, action)
                
                # Skip invalid states
                if not self.problem.is_valid(new_state):
                    continue
                
                # Create a new node for this state
                new_node = Node.Node(new_state, parent=current_node, action=action)
                
                # Calculate g-score (cost from start)
                new_g = current_node.g + self.problem.cost(current_node.state, action)
                
                # Get the state key for checking in open/closed sets
                new_key = self.get_state_key(new_state)
                
                # Skip if we've already processed this state
                if new_key in closed_set:
                    continue
                
                # If this state is already in the open set, check if our new path is better
                if new_key in open_dict:
                    existing_node = open_dict[new_key]
                    if new_g >= existing_node.g:
                        # Our path is not better, skip
                        continue
                
                # This is a better path to the state, update or add to open set
                new_node.g = new_g
                new_node.h = self.get_heuristic(new_state)
                new_node.f = new_node.g + new_node.h
                
                # Add to open set
                heapq.heappush(open_set, (new_node.f, id(new_node), new_node))
                open_dict[new_key] = new_node
        
        self.execution_time = time.time() - start_time
        print(f"ML prediction cache: {self.prediction_hits} hits, {self.prediction_misses} misses")
        
        # If we've examined all nodes and haven't found a solution, return None
        return None
    
    def reconstruct_path(self, goal_node, truncate_to_target_day=True):
        """
        Reconstructs the path from start to goal node.
        
        Args:
            goal_node: The final node to backtrack from
            truncate_to_target_day: Whether to truncate the path to exactly target_day days
            
        Returns:
            A list of actions (intensity, duration) from start to goal
        """
        actions = []
        current = goal_node
        
        # Collect all actions from goal to start
        while current.parent:
            actions.append(current.action)
            current = current.parent
        
        # Reverse to get path from start to goal
        actions = actions[::-1]
        
        # If truncating and we have more actions than target_day, trim the excess
        if truncate_to_target_day and len(actions) > self.problem.target_day:
            actions = actions[:self.problem.target_day]
            print(f"Note: Path truncated from {len(actions)} to {self.problem.target_day} days")
            
        return actions

def test_astar_search(initial_performance=5.0, target_performance=9.0, days=10):
    """
    Test the A* search algorithm with specified parameters.
    
    Args:
        initial_performance: Starting performance level (0-10)
        target_performance: Target performance to achieve (0-10)
        days: Number of days for the training plan
    
    Returns:
        None - prints results to console
    """
    from Problem import AthletePerformanceProblem
    
    print(f"Testing A* Search with:")
    print(f"- Initial Performance: {initial_performance}")
    print(f"- Target Performance: {target_performance}")
    print(f"- Days: {days}")
    print("-" * 50)
    
    # Create the problem with specific state
    problem = AthletePerformanceProblem(
        initial_state=(0, 1.0, 0.1, initial_performance)
    )
    problem.target_day = days
    problem.target_perf = target_performance
    problem.max_fatigue = 3.5  # Maximum acceptable fatigue
    problem.max_risk = 0.2     # Maximum acceptable injury risk
    
    # Create and run the A* search with exact day matching
    astar = AStarSearch(problem)
    goal_node = astar.search(max_iterations=50000, time_limit=180, exact_days=True)
    
    if goal_node:
        # Get the path of actions with truncation to ensure exact day count
        actions = astar.reconstruct_path(goal_node, truncate_to_target_day=True)
        
        print("\nTraining Plan Found:")
        print("-" * 50)
        print("Day | Intensity | Duration | Description")
        print("-" * 50)
        
        # Display initial state
        day, fatigue, risk, performance, _ = problem.initial_state
        print(f"{day:3d} | {'-':9} | {'-':8} | Starting State (P={performance:.1f}, F={fatigue:.1f}, R={risk:.2f})")
        
        # Display each day in the training plan and simulate results
        current_state = problem.initial_state
        for i, action in enumerate(actions):
            intensity, duration = action
            
            # Apply action to get the new state
            current_state = problem.apply_action(current_state, action)
            day, fatigue, risk, performance, _ = current_state
            
            # Get description based on intensity
            if intensity == 0:
                desc = "Rest Day"
            elif intensity <= 0.3:
                desc = "Light Training"
            elif intensity <= 0.6:
                desc = "Moderate Training"
            else:
                desc = "Intense Training"
                
            print(f"{day:3d} | {intensity:9.1f} | {duration:8.0f} | {desc}")
        
        # Print final state
        print("-" * 50)
        print(f"Final State (Day {day}):")
        print(f"Performance: {performance:.2f}")
        print(f"Fatigue: {fatigue:.2f}")
        print(f"Risk: {risk:.2f}")
        print(f"\nA* Search Statistics:")
        print(f"Nodes expanded: {astar.expanded_nodes}")
        print(f"Max queue size: {astar.max_queue_size}")
        print(f"Execution time: {astar.execution_time:.2f} seconds")
        print(f"ML prediction cache: {astar.prediction_hits} hits, {astar.prediction_misses} misses")
    else:
        print("\nNo solution found within iteration/time limit.")
        print(f"Nodes expanded: {astar.expanded_nodes}")
        print(f"Max queue size: {astar.max_queue_size}")
        print(f"Execution time: {astar.execution_time:.2f} seconds")
        print(f"ML prediction cache: {astar.prediction_hits} hits, {astar.prediction_misses} misses")

if __name__ == "__main__":
    # Example usage
    test_astar_search(initial_performance=5.0, target_performance=8, days=10)
