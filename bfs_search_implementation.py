import queue
from collections import deque
import joblib
import numpy as np
import pandas as pd
import functools
from calculate_load_per_minute import calculate_load_per_minute

class Node:
    """
    A node in the search tree representing a state in the search space.
    
    Each node contains a state (day, fatigue, risk, performance, history),
    a reference to its parent node, the action that led to this state,
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

class AthletePerformanceProblem:
    """
    Defines a search problem for athlete performance planning using machine learning models.
    
    This class represents the problem of optimizing a training schedule for an athlete
    over a period of time to maximize performance while managing fatigue and injury risk.
    The problem uses ML models to predict how different training intensities and durations
    affect an athlete's fatigue, performance, and injury risk.
    """
    def __init__(self,
                 initial_state: tuple = (0, 0.0, 0.0, 1.0),
                 target_day: int = 30,
                 target_perf: float = 8.0,
                 max_fatigue: float = 3.0,
                 max_risk: float = 0.5):
        """
        Initialize the athlete performance planning problem.
        
        Args:
            initial_state: Tuple of (day, fatigue, risk, performance)
            target_day: Target day to reach in the training plan
            target_perf: Target performance level (0-10 scale)
            max_fatigue: Maximum allowable fatigue level (0-5 scale)
            max_risk: Maximum allowable injury risk (0-1 probability)
        """
        # Load ML models for predicting fatigue, performance changes and injury risk
        self.delta_f = joblib.load("predictingModels/delta_f_model.pkl")
        self.delta_p = joblib.load("predictingModels/delta_p_model.pkl")
        r_loaded = joblib.load("predictingModels/delta_r_classifier.pkl")
        
        # Extract classifier and feature names from loaded model
        if hasattr(r_loaded, 'predict_proba') and hasattr(r_loaded, 'feature_names_in_'):
            self.delta_r = r_loaded
            self.r_feats = list(r_loaded.feature_names_in_)
        elif isinstance(r_loaded, dict):
            for v in r_loaded.values():
                if hasattr(v, 'predict_proba') and hasattr(v, 'feature_names_in_'):
                    self.delta_r = v
                if isinstance(v, (list, tuple)):
                    self.r_feats = list(v)
        else:
            raise ValueError("Unable to extract injury classifier and features")
            
        # Get intensity-to-load mapping
        self.LOAD_PER_MIN = calculate_load_per_minute()
        
        # Default parameters for athlete's environment
        self.SLEEP_DUR = 7.5  # Average sleep duration in hours
        self.SLEEP_QLT = 3.0  # Sleep quality on scale 1-5
        self.STRESS = 2.5     # Stress level on scale 1-5
        
        # Problem goals and constraints
        self.target_day = target_day
        self.target_perf = target_perf
        self.max_fatigue = max_fatigue
        self.max_risk = max_risk
        
        # Extract feature names from models
        self.f_feats = list(self.delta_f.feature_names_in_) if hasattr(self.delta_f, 'feature_names_in_') else []
        self.p_feats = list(self.delta_p.feature_names_in_) if hasattr(self.delta_p, 'feature_names_in_') else []
        
        # Initialize state with history
        day, f, r, p = initial_state
        self.initial_state = (day, f, r, p, [
            {'load': 0.0,
             'fatigue': f,
             'injury_count': 0,
             'days_since_game': 0,
             'days_since_last_injury': 0}
        ])
        
        # Caches for performance optimization
        self.transition_cache = {}
        self.heuristic_cache = {}
        self.cost_cache = {}

    def state_to_key(self, state):
        """
        Convert a state to a hashable key for caching.
        
        Args:
            state: A tuple containing (day, fatigue, risk, performance, history)
            
        Returns:
            A hashable representation of the state for use in cache dictionaries
        """
        day, fatigue, risk, performance, history = state
        # Convert history to a string representation since it contains dictionaries
        history_str = str(history) if history else "None"
        return (day, fatigue, risk, performance, history_str)
        
    def actions(self, state=None):
        """
        Return available actions for the current state.
        
        Args:
            state: Current state tuple (day, fatigue, risk, performance, history)
            
        Returns:
            List of available actions as (intensity, duration) tuples
        """
        day, fatigue, risk, performance, _ = state if state else (0, 0, 0, 0, None)
        
        # Base actions - combinations of intensity and duration
        all_actions = [(0.0, 0.0),  # Rest day
                (0.3, 60), (0.3, 120),  # Low intensity workouts
                (0.6, 60), (0.6, 120),  # Medium intensity workouts
                (0.9, 60), (0.9, 120)]  # High intensity workouts
        
        # Prune actions if fatigue or risk is high
        if fatigue > self.max_fatigue * 0.8 or risk > self.max_risk * 0.8:
            return [(0.0, 0.0)]  # Only rest if approaching limits

        return all_actions
        
    @functools.lru_cache(maxsize=10000)
    def apply_action(self, state_key, action):
        """
        Apply an action to a state and return the resulting state.
        
        Args:
            state_key: A hashable representation of the state
            action: Tuple of (intensity, duration)
            
        Returns:
            The resulting new state after applying the action
        """
        # Check if result is in cache
        cache_key = (state_key, action)
        if cache_key in self.transition_cache:
            return self.transition_cache[cache_key]
            
        # Convert state_key back to state
        day, F, R, P, history_str = state_key
        # Convert history string back to actual history
        history = eval(history_str) if history_str != "None" else []
        
        # Actual state
        state = (day, F, R, P, history)
        
        # Unpack action
        intensity, duration = action
        is_rest = (intensity == 0.0 and duration == 0.0)
        
        # Compute training load based on intensity and duration
        load = 0.0
        if not is_rest:
            load = self.LOAD_PER_MIN.get(intensity, 0.0) * duration
            
        # Calculate rolling averages for the past 7 days (for features)
        last7 = history[-7:] if len(history) >= 7 else history
        load7 = np.mean([h['load'] for h in last7] + [load])
        fat7 = np.mean([h['fatigue'] for h in last7] + [F])
        
        # Get previous day's state
        prev = history[-1] if history else {'load': 0.0, 'fatigue': F, 'injury_count': 0, 
                                         'days_since_game': 0, 'days_since_last_injury': 0}
        inj_lag1 = int(prev['injury_count'] > 0)
        
        # Handle rest days with simplified calculation (no ML prediction needed)
        if is_rest:
            # Rest day effect: reduce risk by 20%, fatigue by 15%, performance by 4%
            Rn = np.clip(R * 0.8, 0.0, 1.0)  # Risk decreases on rest days
            Fn = max(F * 0.85, 0.0)          # Fatigue decreases on rest days
            Pn = max(P * 0.96, 0.0)          # Performance slightly decreases with no training
        else:
            # For training days, use ML models to predict state changes
            # Assemble feature vector for ML models
            feat = {
                # Current training features
                'load': load,
                'action_intensity': intensity,
                'fatigue_post': F,
                'performance_lag_1': P,
                'sleep_duration': self.SLEEP_DUR,
                'sleep_quality': self.SLEEP_QLT,
                'stress': self.STRESS,
                'is_rest_day': 0,
                'injury_flag_lag_1': inj_lag1,
                
                # Rolling average features
                'load_rolling_7': load7,
                'fatigue_post_rolling_7': fat7,
                'sleep_duration_rolling_7': self.SLEEP_DUR,
                'sleep_quality_rolling_7': self.SLEEP_QLT,
                'stress_rolling_7': self.STRESS,
                
                # History features
                'load_lag_1': prev['load'],
                'total_duration': duration,
                'injury_count': prev['injury_count'],
                'days_since_game': prev['days_since_game'] + 1,
                'days_since_last_injury': prev['days_since_last_injury'] + 1
            }
            X = pd.DataFrame([feat])
            
            # Make predictions using the ML models
            dF = float(self.delta_f.predict(X[self.f_feats])[0])  # Fatigue change
            dP = float(self.delta_p.predict(X[self.p_feats])[0])  # Performance change
            prob = self.delta_r.predict_proba(X[self.r_feats])[0, 1]  # Injury risk probability
            
            # Update state values with predicted changes
            Rn = np.clip(R + prob, 0.0, 1.0)       # New risk
            Fn = np.clip(F + dF, 0.0, 5.0)         # New fatigue
            Pn = max(P + dP, 0.0)                  # New performance

        # Create new history record
        new_rec = {
            'load': load,
            'fatigue': Fn,
            'injury_count': prev['injury_count'],
            'days_since_game': prev['days_since_game'] + 1,
            'days_since_last_injury': prev['days_since_last_injury'] + 1
        }
        
        # Only keep the last 10 days of history to save memory
        new_history = (history + [new_rec])[-10:]
        
        # Create the new state
        new_state = (day + 1, Fn, Rn, Pn, new_history)
        
        # Store in cache and return
        self.transition_cache[cache_key] = new_state
        return new_state

    def is_valid(self, state):
        """
        Check if a state is valid based on fatigue and risk constraints.
        
        A state is valid if both fatigue and risk are below their maximum
        allowable thresholds.
        
        Args:
            state: The state to check
            
        Returns:
            Boolean indicating whether the state is valid
        """
        _, fatigue, risk, _, _ = state
        return fatigue <= self.max_fatigue and risk <= self.max_risk

    def is_goal(self, state):
        """
        Check if a state meets the goal criteria.
        
        A state is considered a goal if it's at or past the target day
        and has reached or exceeded the target performance level.
        
        Args:
            state: The state to check
            
        Returns:
            Boolean indicating whether the state is a goal state
        """
        day, _, _, performance, _ = state
        return day >= self.target_day and performance >= self.target_perf

class BFSSearch:
    """
    Implementation of Breadth-First Search algorithm for athlete training plans.
    
    This class performs a breadth-first search to find an optimal training plan
    by exploring all nodes at the current depth before moving to nodes at the next depth.
    This ensures the shortest path (in terms of number of actions) is found.
    
    Attributes:
        problem: The AthletePerformanceProblem instance
        expanded_nodes: Counter of nodes expanded during search
        max_queue_size: Maximum size of the frontier queue during search
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
        
    def search(self, max_depth=float('inf')):
        """
        Perform breadth-first search to find an optimal training plan.
        
        This function implements a breadth-first search algorithm that uses
        a queue to explore all nodes at the current depth before moving to
        nodes at the next depth. The search continues until a goal state is found
        or the entire search space is explored.
        
        Args:
            max_depth: Maximum search depth (default: infinity)
            
        Returns:
            The goal node if found, None otherwise
        """
        start_node = Node(state=self.problem.initial_state)
        
        # Use deque for more efficient BFS queue
        frontier = deque([start_node])
        
        # Track explored states to avoid cycles
        explored = set()
        
        while frontier:
            # Get next node to explore (FIFO for BFS)
            current_node = frontier.popleft()
            
            # If we've reached the target day with enough performance, return this node
            if self.problem.is_goal(current_node.state):
                return current_node
                
            # Skip already explored states
            rounded_state = self._round_state(current_node.state)
            if rounded_state in explored:
                continue
            
            # Mark this state as explored
            explored.add(rounded_state)
            
            # Get the valid actions from the current state
            for action in self.problem.actions(current_node.state):
                # Convert state to a key for caching
                state_key = self.problem.state_to_key(current_node.state)
                
                # Apply the action to get a new state
                new_state = self.problem.apply_action(state_key, action)
                
                # Skip invalid states
                if not self.problem.is_valid(new_state):
                    continue
                
                # Create a new node for this state
                child_node = Node(new_state, parent=current_node, action=action)
                
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
        
    def _round_state(self, state):
        """
        Round state values to reduce the state space and avoid similar states.
        
        Args:
            state: The state to round
            
        Returns:
            A hashable tuple with rounded state values
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
        return path[::-1]  # reverse to get the correct order

def test_bfs_search():
    """
    Test the BFS algorithm with predefined parameters.
    
    This function creates an athlete performance planning problem with
    specific initial state and target values, then runs the BFS algorithm
    to find a training plan that achieves the goals. The resulting
    plan is displayed with details for each day.
    
    Returns:
        None
    """
    print("Testing BFS Algorithm")
    print("-----------------------------------------")
    
    # Create the athlete performance problem with specific parameters
    problem = AthletePerformanceProblem(
        initial_state=(0, 1.5, 0.2, 6.0),  # Initial state: day 0, fatigue 1.5, risk 0.2, performance 6.0
        target_day=5,                      # Training plan duration (shorter for BFS)
        target_perf=6.5,                    # Target performance level
        max_fatigue=4,                    # Maximum allowable fatigue
        max_risk=0.4                        # Maximum allowable injury risk
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
            state_key = problem.state_to_key(state)
            state = problem.apply_action(state_key, action)
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
        if problem.is_goal(state):
            print("\nGoal achieved!")
        else:
            print("\nGoal not achieved.")

# Run the test if this file is executed directly
if __name__ == "__main__":
    test_bfs_search() 