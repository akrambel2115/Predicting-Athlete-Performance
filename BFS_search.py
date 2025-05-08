from collections import deque
from Problem import AthletePerformanceProblem
from Node import Node

class BFSSearch:
    """
    Implementation of Breadth-First Search algorithm for athlete training plans.
    
    This class performs a breadth-first search to find an optimal training plan
    by exploring all nodes at the current depth before moving to nodes at the next depth.
    This ensures the shortest path (in terms of number of actions) is found.
    """
    def __init__(self, problem):
        """
        Initialize the BFS search algorithm with a problem instance.
        
        Args:
            problem: An AthletePerformanceProblem instance
        """
        self.problem = problem
        self.expanded_nodes = 0
        self.max_queue_size = 0
        
        # Set target day and performance (needed for is_goal)
        self.problem.target_day = 10
        self.problem.target_perf = 6.5
        self.problem.max_fatigue = 2.7
        self.problem.max_risk = 0.3
        
    def search(self, max_depth=float('inf')):
        """
        Perform breadth-first search to find an optimal training plan.
        """
        start_node = Node(state=self.problem.initial_state, costless=True)
        
        # Use deque for more efficient BFS queue
        frontier = deque([start_node])
        
        # Track explored states to avoid cycles
        explored = set()
        
        while frontier:
            # Get next node to explore (FIFO for BFS)
            current_node = frontier.popleft()
            
            day, _, _, performance, _ = current_node.state
            # Check if goal state (using custom goal check)
            if day >= self.problem.target_day and performance >= self.problem.target_perf:
                return current_node
                
            # Skip already explored states
            rounded_state = self._round_state(current_node.state)
            if rounded_state in explored:
                continue
            
            # Mark this state as explored
            explored.add(rounded_state)
            
            # Get the valid actions from the current state
            for action in self.problem.actions():
                # Apply the action to get a new state
                # Create a state that includes history, as expected by the updated apply_action method
                current_state = current_node.state  # Full state with history
                new_state = self.problem.apply_action(current_state, action)
                
                # Skip invalid states
                if not self.is_valid(new_state):
                    continue
                
                # Create a new node for this state
                child_node = Node(new_state, parent=current_node, action=action, costless=True)
                
                # Skip if exceeds maximum depth
                if child_node.depth > max_depth:
                    continue
                    
                # Add to frontier for further exploration
                frontier.append(child_node)
            
            self.expanded_nodes += 1
            
            # Track maximum queue size
            self.max_queue_size = max(self.max_queue_size, len(frontier))
            
            # Progress indicator
            if self.expanded_nodes % 500 == 0:
                print(f"Explored {self.expanded_nodes} nodes, queue size: {len(frontier)}")
            
        # If we've examined all nodes and haven't found a solution, return None
        return None
    
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
        """
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return path[::-1]  # reverse to get the correct order

def test_bfs_search():
    """
    Test the BFS algorithm with predefined parameters.
    """
    print("Testing BFS Algorithm")
    print("-----------------------------------------")
    
    # Create the athlete performance problem with specific parameters
    # We'll set target_day, target_perf, etc. inside the BFSSearch class
    problem = AthletePerformanceProblem(
        initial_state=(0, 1.5, 0.2, 6.0)  # Initial state: day 0, fatigue 1.5, risk 0.2, performance 6.0
    )
    
    # Create the BFS algorithm
    searcher = BFSSearch(problem)

    # Run the search algorithm
    print("Starting search...")
    goal_node = searcher.search()
    print(f"Search completed. Nodes explored: {searcher.expanded_nodes}")
    
    # Report the results
    if goal_node is None:
        print("No solution found.")
    else:
        # Reconstruct the path from initial state to goal
        path = searcher.reconstruct_path(goal_node)
        
        # Display the training plan as a table
        print("\nTraining Plan:")
        print("Day | Intensity | Duration | Fatigue | Risk | Performance")
        print("----|-----------|----------|---------|------|------------")
        
        # Display initial state
        state = problem.initial_state
        day = 0
        print(f"{day:3d} | {'-':9} | {'-':8} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")
        
        # Display each day in the training plan
        for action in path:
            # Apply the action using the updated method signature
            state = problem.apply_action(state, action)
            day += 1
            intensity, duration = action
            print(f"{day:3d} | {intensity:9.1f} | {duration:8.1f} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")
        
        # Display final state summary
        final_day, final_fatigue, final_risk, final_perf, _ = state
        print("\nFinal State:")
        print(f"Day: {final_day}")
        print(f"Fatigue: {final_fatigue:.2f}/5.00")
        print(f"Risk: {final_risk:.2f}/1.00")
        print(f"Performance: {final_perf:.2f}/10.00")
        
        # Report whether goal was achieved
        if final_day >= searcher.problem.target_day and final_perf >= searcher.problem.target_perf:
            print("\nGoal achieved!")
        else:
            print("\nGoal not achieved.")

# Run the test if this file is executed directly
if __name__ == "__main__":
    import time
    
    # Start timer
    start_time = time.time()
    
    # Run the BFS search
    test_bfs_search()
    
    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
