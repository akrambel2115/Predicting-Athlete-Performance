import queue
import time
from Node import Node
from Problem import AthletePerformanceProblem

class UCSSearch:
    """
    Implementation of Uniform Cost Search algorithm for athlete training plans.
    
    This class performs a uniform-cost search to find an optimal training plan
    by exploring nodes with the lowest path cost (g-value) first. This guarantees
    finding the lowest-cost path to the goal state, if one exists.
    
    Attributes:
        problem: The AthletePerformanceProblem instance
        expanded_nodes: Counter of nodes expanded during search
        max_queue_size: Maximum size of the frontier queue during search
        execution_time: Time taken to execute the search
    """
    def __init__(self, problem):
        """
        Initialize the UCS search algorithm with a problem instance.
        
        Args:
            problem: An AthletePerformanceProblem instance
        """
        self.problem = problem
        self.expanded_nodes = 0
        self.max_queue_size = 0
        self.execution_time = 0
        
        # Set target day and performance
        self.problem.target_day = 10
        self.problem.target_perf = 7
        self.problem.max_fatigue = 3.5
        self.problem.max_risk = 2.5
        
    def search(self, max_depth=10):
        """
        Performs a uniform-cost search to find an optimal training plan.
        
        This method expands nodes with the lowest cumulative path cost (g-value) first,
        guaranteeing the optimal solution if one exists. It uses the problem's cost
        function to calculate the cost of each action.
        
        Args:
            max_depth: Maximum depth to explore in the search tree
            
        Returns:
            The goal node if a solution is found, None otherwise
        """
        start_time = time.time()
        
        # Create initial node from the problem's initial state
        initial_node = Node(self.problem.initial_state, g=0)
        
        # Priority queue for UCS - priority is cumulative path cost (g-value)
        frontier = queue.PriorityQueue()
        frontier.put((0, initial_node))  # Initial cost is 0
        
        # Track explored states and their costs
        explored = {}  # Maps rounded state to lowest g-value found
        
        # Best solution found so far
        best_solution = None
        best_cost = float('inf')
        
        while not frontier.empty():
            # Get node with lowest path cost
            current_cost, current_node = frontier.get()
            
            day, fatigue, risk, performance, _ = current_node.state
            
            # Check if we've reached the target day
            if day == self.problem.target_day and performance >= self.problem.target_perf:
                if current_cost < best_cost:
                    best_solution = current_node
                    best_cost = current_cost
                    
                    # Since UCS guarantees the optimal path to any node,
                    # if we want the first solution at target day, we can return it
                    self.execution_time = time.time() - start_time
                    return best_solution
            
            # Don't explore beyond target day
            if day > self.problem.target_day:
                continue
                
            # Check if we've already found a better path to this state
            rounded_state = self._round_state(current_node.state)
            if rounded_state in explored and explored[rounded_state] <= current_cost:
                continue
            
            # Remember this as the best path to this state so far
            explored[rounded_state] = current_cost
            
            # Skip if exceeds maximum depth
            if current_node.depth >= max_depth:
                continue
                
            # Get valid actions for current state
            for action in self.problem.actions():
                # Apply action to get new state
                new_state = self.problem.apply_action(current_node.state, action)
                
                # Skip invalid states
                if not self.is_valid(new_state):
                    continue
                
                # Calculate action cost
                action_cost = self.problem.cost(current_node.state, action)
                
                # Calculate cumulative path cost
                new_cost = current_node.g + action_cost
                
                # Create new node with updated cost
                child_node = Node(
                    state=new_state, 
                    parent=current_node, 
                    action=action, 
                    g=new_cost, 
                    f=new_cost  # In UCS, f = g (no heuristic)
                )
                
                # Add to frontier with priority based on path cost
                frontier.put((new_cost, child_node))
            
            self.expanded_nodes += 1
            
            # Track maximum queue size
            self.max_queue_size = max(self.max_queue_size, frontier.qsize())
            
            # Progress indicator
            if self.expanded_nodes % 200 == 0:
                print(f"Explored {self.expanded_nodes} nodes, queue size: {frontier.qsize()}")
        
        self.execution_time = time.time() - start_time
        return best_solution
    
    def is_valid(self, state):
        """
        Check if a state is valid based on constraints.
        
        Args:
            state: The state to validate
            
        Returns:
            True if state satisfies all constraints, False otherwise
        """
        _, fatigue, risk, _, _ = state
        return fatigue <= self.problem.max_fatigue and risk <= self.problem.max_risk
        
    def _round_state(self, state):
        """
        Round state values to reduce the state space and avoid similar states.
        
        Args:
            state: The state to round
            
        Returns:
            A tuple with rounded values (excluding history)
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
        
        Args:
            node: The goal node
            
        Returns:
            A list of actions from start to goal in correct order
        """
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return path[::-1]  # Reverse to get the correct order

def test_ucs_search():
    """
    Test the UCS algorithm with predefined parameters.
    
    This function creates an athlete performance planning problem with
    specific initial state and target values, then runs the UCS search
    algorithm to find an optimal training plan.
    
    Returns:
        None
    """
    print("Testing Uniform Cost Search Algorithm")
    print("-----------------------------------------")
    
    # Create the athlete performance problem with specific parameters
    problem = AthletePerformanceProblem(
        initial_state=(0, 1.5, 0.2, 6.0)  # Initial state (day, fatigue, risk, performance)
    )
    
    # Create the searcher
    searcher = UCSSearch(problem)
    
    # Run the search algorithm
    print("Starting search...")
    goal_node = searcher.search()
    print(f"Search completed. Nodes explored: {searcher.expanded_nodes}")
    
    if goal_node is None:
        print("No solution found.")
    else:
        path = searcher.reconstruct_path(goal_node)
        print("\nTraining Plan:")
        print("Day | Intensity | Duration | Fatigue | Risk | Performance | Cumulative Cost")
        print("----|-----------|----------|---------|------|-------------|----------------")
        
        # Display initial state
        state = problem.initial_state
        day, fatigue, risk, performance, _ = state
        print(f"{day:3d} | {'-':9} | {'-':8} |  {fatigue:.2f}   | {risk:.2f} | {performance:.2f}      | 0.00")
        
        # Display each day in the training plan
        current_state = state
        cumulative_cost = 0
        for i, action in enumerate(path):
            # Calculate cost of this action
            action_cost = problem.cost(current_state, action)
            cumulative_cost += action_cost
            
            # Apply action to get the new state
            current_state = problem.apply_action(current_state, action)
            
            # Unpack the new state for display
            day, fatigue, risk, performance, _ = current_state
            intensity, duration = action
            print(f"{day:3d} | {intensity:9.1f} | {duration:8.1f} |  {fatigue:.2f}   | {risk:.2f} | {performance:.2f}      | {cumulative_cost:.2f}")
        
        # Display final state summary
        final_day, final_fatigue, final_risk, final_perf, _ = current_state
        print("\nFinal State:")
        print(f"Day: {final_day}")
        print(f"Fatigue: {final_fatigue:.2f}/5.00")
        print(f"Risk: {final_risk:.2f}/1.00")
        print(f"Performance: {final_perf:.2f}/10.00")
        print(f"Total Path Cost: {goal_node.g:.2f}")
        print(f"Execution Time: {searcher.execution_time:.2f} seconds")
        print(f"Expanded Nodes: {searcher.expanded_nodes}")
        print(f"Max Queue Size: {searcher.max_queue_size}")
        
        # Report whether goal was achieved
        if final_day == searcher.problem.target_day and final_perf >= searcher.problem.target_perf:
            print("\nGoal achieved with optimal cost!")
        else:
            print("\nGoal not achieved.")

if __name__ == "__main__":
    test_ucs_search()