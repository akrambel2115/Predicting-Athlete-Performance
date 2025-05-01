import queue
from collections import deque
import numpy as np
import pandas as pd
from Problem import AthletePerformanceProblem

class Node:
    """
    A node in the search tree representing a state in the search space.
    
    Each node contains a state (day, fatigue, risk, performance, history),
    a reference to its parent node, the action that led to this state,k
    and the depth in the search tree.
    """
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = 0 if parent is None else parent.depth + 1

    def __hash__(self):
        return hash(self.state[:4])  # Only hash the numeric part of state

    def __eq__(self, other):
        return isinstance(other, Node) and self.state[:4] == other.state[:4]

# Fix the variable naming issue in Problem.py by monkey patching 
# the apply_action method to handle the naming inconsistency
original_apply_action = AthletePerformanceProblem.apply_action

def fixed_apply_action(self, state, action, history):
    """Fixed version of apply_action that handles variable naming properly"""
    # Unpack
    day, F, R, P = state
    intensity, duration = action
    is_rest = (intensity == 0.0 and duration == 0.0)
    # Compute load
    load = 0.0
    if not is_rest:
        load = self.LOAD_PER_MIN.get(intensity, 0.0) * duration
    # Rolling-7 calculations
    last7 = history[-7:]
    load7 = np.mean([h['load'] for h in last7] + [load])
    fat7  = np.mean([h['fatigue'] for h in last7] + [F])
    prev = history[-1]
    inj_lag1 = int(prev['injury_count'] > 0)
    # Assemble features
    feat = {
        'load': load,
        'action_intensity': intensity,
        'fatigue_post': F,
        'performance_lag_1': P,
        'sleep_duration': self.SLEEP_DUR,
        'sleep_quality':  self.SLEEP_QLT,
        'stress':         self.STRESS,
        'is_rest_day':    int(is_rest),
        'injury_flag_lag_1': inj_lag1,
        'load_rolling_7':      load7,
        'fatigue_post_rolling_7': fat7,
        'sleep_duration_rolling_7': self.SLEEP_DUR,
        'sleep_quality_rolling_7':  self.SLEEP_QLT,
        'stress_rolling_7':        self.STRESS,
        'load_lag_1':      prev['load'],
        'total_duration':  duration,
        'injury_count':    prev['injury_count'],
        'days_since_game': prev['days_since_game'] + 1,
        'days_since_last_injury': prev['days_since_last_injury'] + 1
    }
    X = pd.DataFrame([feat])
    # Predictions
    dF = float(self.delta_f.predict(X[self.f_feats])[0])
    dP = float(self.delta_p.predict(X[self.p_feats])[0])
    if is_rest:
        Rn = np.clip(R * 0.8, 0.0, 1.0)
        # Fix variable names for rest case
        Fn = max(F * 0.85, 0.0)  # Changed from F_new to Fn
        Pn = max(P * 0.96, 0.0)  # Changed from P_new to Pn
    else:
        prob = self.delta_r.predict_proba(X[self.r_feats])[0, 1]
        Rn = np.clip(R + prob, 0.0, 1.0)
        Fn = np.clip(F + dF, 0.0, 5.0)
        Pn = max(P + dP, 0.0)

    # Update history
    new_rec = {
        'load': load,
        'fatigue': Fn,
        'injury_count': prev['injury_count'],
        'days_since_game': feat['days_since_game'],
        'days_since_last_injury': feat['days_since_last_injury']
    }
    new_history = history + [new_rec]
    return (day + 1, Fn, Rn, Pn, new_history)

# Replace the original method with our fixed version
AthletePerformanceProblem.apply_action = fixed_apply_action

class DFSSearch:
    """
    Implementation of Depth-First Search algorithm for athlete training plans.
    
    This class performs a depth-first search to find a training plan
    by exploring as far as possible along each branch before backtracking.
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
        self.problem.target_day = 30
        self.problem.target_perf = 8
        self.problem.max_fatigue = 2
        self.problem.max_risk = 0.2
        
    def search(self, max_depth=float('inf')):
        """
        Perform depth-first search to find an optimal training plan.
        """
        start_node = Node(state=self.problem.initial_state)
        
        # Use a list as stack for DFS
        frontier = [start_node]
        
        # Track explored states to avoid cycles
        explored = set()
        
        while frontier:
            # Get next node to explore (LIFO for DFS)
            current_node = frontier.pop()
            
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
            
            # Skip if exceeds maximum depth
            if current_node.depth >= max_depth:
                continue
                
            # Get the valid actions from the current state
            # Reverse actions to maintain the original order when popping from stack
            for action in reversed(self.problem.actions()):
                # Apply the action to get a new state
                state_tuple = current_node.state[:4]  # Extract (day, fatigue, risk, performance)
                history = current_node.state[4]       # Extract history
                new_state = self.problem.apply_action(state_tuple, action, history)
                
                # Skip invalid states
                if not self.is_valid(new_state):
                    continue
                
                # Create a new node for this state
                child_node = Node(new_state, parent=current_node, action=action)
                
                # Add to frontier for further exploration
                frontier.append(child_node)
            
            self.expanded_nodes += 1
            
            # Track maximum stack size
            self.max_stack_size = max(self.max_stack_size, len(frontier))
            
            # Progress indicator
            if self.expanded_nodes % 500 == 0:
                print(f"Explored {self.expanded_nodes} nodes, stack size: {len(frontier)}")
            
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

def test_dfs_search():
    """
    Test the DFS algorithm with predefined parameters.
    """
    print("Testing DFS Algorithm")
    print("-----------------------------------------")
    
    # Create the athlete performance problem with specific parameters
    # We'll set target_day, target_perf, etc. inside the DFSSearch class
    problem = AthletePerformanceProblem(
        initial_state=(0, 1.5, 0.2, 6.0)  # Initial state: day 0, fatigue 1.5, risk 0.2, performance 6.0
    )
    
    # Create the DFS algorithm
    searcher = DFSSearch(problem)

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
            state_tuple = state[:4]  # Extract (day, fatigue, risk, performance)
            history = state[4]       # Extract history
            state = problem.apply_action(state_tuple, action, history)
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
    test_dfs_search() 