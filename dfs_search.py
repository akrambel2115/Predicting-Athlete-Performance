from collections import deque
from Problem import AthletePerformanceProblem
from Node import Node

class DFSSearch:
    """
    Implementation of Depth-First Search algorithm for athlete training plans.
    
    This class performs a depth-first search to find a training plan
    by exploring as far as possible along each branch before backtracking.
    DFS can find solutions quickly in some cases but does not guarantee optimality.
    """
    def __init__(self, problem):
        """
        Initialize the DFS search algorithm with a problem instance.
        
        Args:
            problem: An AthletePerformanceProblem instance
        """
        self.problem = problem
        self.expanded_nodes = 0
        self.max_stack_size = 0

        # Set target day and performance (needed for is_goal)
        self.problem.target_day = 10
        self.problem.target_perf = 7
        self.problem.max_fatigue = 4
        self.problem.max_risk = 0.3

    def search(self, max_depth=10):
        """
        Perform depth-first search to find a training plan.
        
        Args:
            max_depth: Maximum depth to explore in the search tree
        
        Returns:
            The goal node if a solution is found, None otherwise
        """
        start_node = Node(state=self.problem.initial_state, costless=True)
        # Use stack (LIFO) for DFS
        stack = deque([start_node])
        # Track explored states to avoid cycles
        explored = set()

        while stack:
            # Get next node to explore (LIFO for DFS)
            current_node = stack.pop()

            day, fatigue, risk, performance, _ = current_node.state
            # Check if goal state (using custom goal check)
            if day >= self.problem.target_day and performance >= self.problem.target_perf:
                print(f"Goal found! Day: {day}, Performance: {performance:.2f}, Fatigue: {fatigue:.2f}, Risk: {risk:.2f}")
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
                current_state = current_node.state
                new_state = self.problem.apply_action(current_state, action)

                # Skip invalid states
                if not self.is_valid(new_state):
                    continue

                # Create a new node for this state
                child_node = Node(new_state, parent=current_node, action=action, costless=True)

                # Skip if exceeds maximum depth
                if child_node.depth > max_depth:
                    continue

                # Add to stack for further exploration
                stack.append(child_node)

            self.expanded_nodes += 1
            # Track maximum stack size
            self.max_stack_size = max(self.max_stack_size, len(stack))
            
            # Progress indicator
            if self.expanded_nodes % 100 == 0:
                print(f"Explored {self.expanded_nodes} nodes, stack size: {len(stack)}, " 
                      f"Current state: Day {day}, F={fatigue:.2f}, R={risk:.2f}, P={performance:.2f}, Depth={current_node.depth}")

        # If we've examined all nodes and haven't found a solution, return None
        return None

    def is_valid(self, state):
        """
        Check if a state is valid based on constraints.
        
        Args:
            state: The state to check
            
        Returns:
            True if the state is valid, False otherwise
        """
        _, fatigue, risk, _, _ = state
        return fatigue <= self.problem.max_fatigue and risk <= self.problem.max_risk

    def _round_state(self, state):
        """
        Round state values to reduce the state space and avoid similar states.
        
        Args:
            state: The state to round
            
        Returns:
            A tuple with rounded state values
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
            A list of actions that lead from the initial state to the goal state
        """
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return path[::-1]  # reverse to get the correct order

def test_dfs_search():
    print("Testing DFS Algorithm")
    print("-----------------------------------------")

    problem = AthletePerformanceProblem(
        initial_state=(0, 1.5, 0.2, 6.0)
    )

    searcher = DFSSearch(problem)

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

        state = problem.initial_state
        day = 0
        print(f"{day:3d} | {'-':9} | {'-':8} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")

        for action in path:
            state = problem.apply_action(state, action)
            day += 1
            intensity, duration = action
            print(f"{day:3d} | {intensity:9.1f} | {duration:8.1f} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")

        final_day, final_fatigue, final_risk, final_perf, _ = state
        print("\nFinal State:")
        print(f"Day: {final_day}")
        print(f"Fatigue: {final_fatigue:.2f}/5.00")
        print(f"Risk: {final_risk:.2f}/1.00")
        print(f"Performance: {final_perf:.2f}/10.00")

        if final_day >= searcher.problem.target_day and final_perf >= searcher.problem.target_perf:
            print("\nGoal achieved!")
        else:
            print("\nGoal not achieved.")

if __name__ == "__main__":
    import time

    start_time = time.time()
    test_dfs_search()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")