import queue
import time
from Node import Node as Node
from Problem import AthletePerformanceProblem as AthletePerformanceProblem

class GreedySearch:
    """
    Implementation of a greedy best-first search algorithm for athlete training plans.
    
    This class performs a greedy best-first search to find an optimal training plan
    by using a heuristic to guide the search toward promising states. The search
    expands nodes based on their heuristic values, always exploring the most
    promising node first.
    
    Attributes:
        problem: The AthletePerformanceProblem instance
        expanded_nodes: Counter of nodes expanded during search
        max_queue_size: Maximum size of the frontier queue during search
    """
    def __init__(self, problem):
        self.problem = problem
        self.expanded_nodes = 0
        self.max_queue_size = 0
        self.execution_time = 0
        
        # Set target day and performance
        self.problem.target_day = 14
        self.problem.target_perf = 8
        self.problem.max_fatigue = 2.7
        self.problem.max_risk = 0.2
        
    def search(self, max_depth=float('inf')):
        """
        Performs a greedy best-first search to find an optimal training plan.
        
        This method uses generates successor nodes
        and explores them based on their heuristic values (greedy best-first search).
        
        Args:
            max_depth: Maximum depth to explore in the search tree
            
        Returns:
            The goal node if a solution is found, None otherwise
        """
        start_time = time.time()
        
        # Create initial node from the problem's initial state
        initial_node = Node(self.problem.initial_state)
        
        # Priority queue for greedy best-first search
        frontier = queue.PriorityQueue()
        frontier.put((self._get_priority(initial_node), initial_node))
        
        # Track explored states to avoid cycles
        explored = set()
        
        while not frontier.empty():
            # Get next node to explore (with lowest priority/heuristic value)
            _, current_node = frontier.get()
            
            day, _, _, performance, _ = current_node.state
            
            # Check if goal state
            if day == self.problem.target_day and performance >= self.problem.target_perf:
                # Exact day match with performance target - ideal solution
                self.execution_time = time.time() - start_time
                return current_node
                            
            # Skip already explored states
            rounded_state = self._round_state(current_node.state)
            if rounded_state in explored:
                continue
            
            # Mark this state as explored
            explored.add(rounded_state)
            
            # Skip if exceeds maximum depth or target day
            if current_node.depth >= max_depth or day >= self.problem.target_day:
                continue
                
            # Get the valid actions from the current state
            for action in self.problem.actions():
                # Apply action to get a new state
                # Pass the complete state to apply_action
                new_state = self.problem.apply_action(current_node.state, action)
                
                # Skip invalid states
                if not self.is_valid(new_state):
                    continue
                
                # Create a new node for this state
                child_node = Node(new_state, parent=current_node, action=action)
                
                # Add to frontier with priority based on heuristic
                frontier.put((self._get_priority(child_node), child_node))
            
            self.expanded_nodes += 1
            
            # Track maximum queue size
            self.max_queue_size = max(self.max_queue_size, frontier.qsize())
            
            # Progress indicator
            if self.expanded_nodes % 500 == 0:
                print(f"Explored {self.expanded_nodes} nodes, queue size: {frontier.qsize()}")
        
        self.execution_time = time.time() - start_time
                # If we've examined all nodes and haven't found a solution, return None
        return None
    
    def _get_priority(self, node):
        """
        Calculate priority for a node based on problem heuristic.
        Lower values have higher priority in the queue.
        """
        return self.problem.heuristic(node.state)
    
    def is_valid(self, state):
        """Check if a state is valid based on constraints."""
        _, fatigue, risk, _, _ = state
        return fatigue <= self.problem.max_fatigue and risk <= self.problem.max_risk
        
    def _round_state(self, state):
        """
        Round state values to reduce the state space and avoid similar states.
        """
        day, fatigue, risk, performance, _ = state
        return (
            day,
            round(fatigue, 1),  # Round fatigue to 1 decimal place
            round(risk, 1),     # Round risk to 1 decimal place
            round(performance, 1)  # Round performance to 1 decimal place
        )
    
    def reconstruct_path(self, node):
        """
        Reconstruct the path from the initial state to the goal state.
        
        This function traces back from the goal node to the start node
        using parent references, collecting actions along the way.
        
        Args:
            node: The goal node
            
        Returns:
            A list of actions from start to goal in correct order
        """
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return path[::-1]  # reverse to get the correct order

def test_greedy_search():
    """
    Test the greedy search algorithm with predefined parameters.
    
    This function creates an athlete performance planning problem with
    specific initial state and target values, then runs the greedy search
    algorithm to find a training plan that achieves the goals. The resulting
    plan is displayed with details for each day.
    
    Returns:
        None
    """
    print("Testing Greedy Search Algorithm")
    print("-----------------------------------------")
    
    # Create the athlete performance problem with specific parameters
    problem = AthletePerformanceProblem(
        initial_state=(0, 1.5, 0.05, 6)  # Initial state (day, fatigue, risk, performance)
    )
    
    # Create the searcher
    searcher = GreedySearch(problem)
    
    # Run the search algorithm
    print("Starting search...")
    goal_node = searcher.search()
    print(f"Search completed. Nodes explored: {searcher.expanded_nodes}")
    
    if goal_node is None:
        print("No solution found.")
    else:
        path = searcher.reconstruct_path(goal_node)
        print("\nTraining Plan:")
        print("Day | Intensity | Duration | Fatigue | Risk | Performance")
        print("----|-----------|----------|---------|------|------------")
        
        # Display initial state
        state = problem.initial_state
        day, fatigue, risk, performance, history = state
        print(f"{day:3d} | {'-':9} | {'-':8} |  {fatigue:.2f}   | {risk:.2f} | {performance:.2f}")
        
        # Display each day in the training plan
        current_state = state
        for action in path:
            # Apply action to get the new state
            current_state = problem.apply_action(current_state, action)
            # Unpack the new state for display
            day, fatigue, risk, performance, _ = current_state
            intensity, duration = action
            
            # Categorize rest days based on fatigue levels
            if intensity == 0.0 and duration == 0.0:
                # Determine rest type based on fatigue level
                if fatigue > 2.0:
                    rest_type = "Passive Rest"
                else:
                    rest_type = "Active Rest"
                print(f"{day:3d} | {rest_type:9} | {'-':8} |  {fatigue:.2f}   | {risk:.2f} | {performance:.2f}")
            else:
                print(f"{day:3d} | {intensity:9.1f} | {duration:8.1f} |  {fatigue:.2f}   | {risk:.2f} | {performance:.2f}")
        
        # Report whether goal was achieved
        final_day, final_fatigue, final_risk, final_perf, _ = current_state
        if final_day >= searcher.problem.target_day and final_perf >= searcher.problem.target_perf:
            print("\nGoal achieved!")
        else:
            print("\nGoal not achieved, returning next best solution:")
            
        # Display final state summary
        print("\nFinal State:")
        print(f"Day: {final_day}")
        print(f"Fatigue: {final_fatigue:.2f}/4.00")
        print(f"Risk: {final_risk:.2f}/1.00")
        print(f"Performance: {final_perf:.2f}/10.00")
        print(f"Execution Time: {searcher.execution_time:.2f} seconds")
        print(f"Expanded Nodes: {searcher.expanded_nodes}")
        print(f"Max Queue Size: {searcher.max_queue_size}")
        

if __name__ == "__main__":
    test_greedy_search()
