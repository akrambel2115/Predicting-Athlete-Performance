#TODO get better cost and heruistic functions, tune their weights and write the code in the main notebook

import queue
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
    and various cost metrics used by search algorithms.
    
    Attributes:
        state: Tuple containing (day, fatigue, risk, performance, history)
        parent: Reference to the parent Node
        action: The action (intensity, duration) that led to this state
        g: Path cost from start to this node
        f: Total evaluation function value (g + h)
        h: Heuristic value (estimated cost to goal)
        depth: Depth of this node in the search tree
    """
    def __init__(self, state, parent=None, action=None, g=0, f=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.f = f
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
    Defines a search problem for athlete performance planning using machine learning models.
    
    This class represents the problem of optimizing a training schedule for an athlete
    over a period of time to maximize performance while managing fatigue and injury risk.
    The problem uses ML models to predict how different training intensities and durations
    affect an athlete's fatigue, performance, and injury risk.
    
    State representation:
        - day: Current day in the training period
        - fatigue: Current fatigue level (0-5 scale)
        - risk: Current injury risk probability (0-1 scale)
        - performance: Current performance level (0-10 scale)
        - history: List of previous training records
        
    Actions:
        - Train: Tuple of (intensity_value, duration_minutes)
        - Rest: (0.0, 0.0)
        
    Attributes:
        delta_f: ML model for predicting fatigue change
        delta_p: ML model for predicting performance change
        delta_r: ML model for predicting injury risk
        LOAD_PER_MIN: Dictionary mapping intensity to load per minute
        target_day: Target training period length in days
        target_perf: Target performance level to achieve
        max_fatigue: Maximum allowable fatigue level
        max_risk: Maximum allowable injury risk probability
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
                (0.3, 60), (0.3, 90), (0.3, 120),  # Low intensity workouts
                (0.6, 60), (0.6, 90), (0.6, 120),  # Medium intensity workouts
                (0.9, 60), (0.9, 90), (0.9, 120)]  # High intensity workouts
        

            
        return all_actions
        
    @functools.lru_cache(maxsize=10000) # I have no idea what this does, but the AI says to keep it
    def apply_action(self, state_key, action):
        """
        Apply an action to a state and return the resulting state.
        
        This function is the core of the state transition model. It takes an action
        (training intensity and duration) and applies it to the current state,
        using ML models to predict the resulting changes in fatigue, performance,
        and injury risk.
        
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
            Rn = np.clip(R * 0.85, 0.0, 1.0)  # Risk decreases on rest days
            Fn = max(F * 0.85, 0.0)          # Fatigue decreases on rest days
            Pn = max(P * 0.92, 0.0)          # Performance slightly decreases with no training
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
            Pn = np.clip(P + dP, 0. , 10)                  # New performance

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

    def expand_node(self, node, use_cost=False, use_heuristic=False):
        """
        Expands a node by applying all possible actions and returning the resulting nodes.
        
        This function generates all valid successor states by applying each possible
        action to the current state, then creates Node objects for each valid successor.
        
        Args:
            node: The Node to expand
            use_cost: Whether to calculate cost for each child node
            use_heuristic: Whether to calculate heuristic for each child node
            
        Returns:
            List of valid child Nodes resulting from applying actions
        """
        children = []
        state_key = self.state_to_key(node.state)
        current_state = node.state
        
        # Get applicable actions for the current state
        for action in self.actions(node.state):
            # Apply action using the state key for caching
            new_state = self.apply_action(state_key, action)
            
            if self.is_valid(new_state):
                # Calculate transition cost from current_state to new_state via action
                transition_cost = self.cost(new_state, action, current_state) if use_cost else 0
                
                # Calculate total path cost: parent's path cost + this transition cost
                g = node.g + transition_cost
                
                # Calculate heuristic value if requested
                heuristic = self.heuristic(new_state) if use_heuristic else 0
                
                # Total evaluation function value
                f = g + heuristic
                
                # Create new node with updated values
                new_node = Node(
                    state=new_state,
                    parent=node,
                    action=action,
                    g=g,
                    f=f,
                    h=heuristic
                )
                
                children.append(new_node)
                
        return children

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
    
    def cost(self, state, action=None, prev_state=None):
        """
        Calculate the cost of transitioning from prev_state to state via action.
        
        This function evaluates the cost of making a transition from one state to
        another through a specific action. It prioritizes:
        1. Training efficiency (high performance gain relative to increased fatigue/risk)
        2. Risk management (avoiding risky transitions)
        3. Recovery optimization (when resting)
        
        Args:
            state: The resulting state after the action
            action: The (intensity, duration) action taken
            prev_state: The previous state before the action (if applicable)
            
        Returns:
            A numerical cost value for this transition (lower is better)
        """
        cache_key = (self.state_to_key(state), 
                    str(action), 
                    self.state_to_key(prev_state) if prev_state else None)
                    
        if cache_key in self.cost_cache:
            return self.cost_cache[cache_key]
            
        # If no previous state or action provided, fall back to state evaluation
        if prev_state is None:
            day, fatigue, risk, performance, _ = state
            
            # Performance deficit from target
            perf_deficit = max(0, self.target_perf - performance)
            risk_factor = risk / self.max_risk if self.max_risk > 0 else 0
            fatigue_factor = fatigue / self.max_fatigue if self.max_fatigue > 0 else 0
            
            cost = (
                5.0 * perf_deficit + 
                2.0 * risk_factor + 
                1.5 * fatigue_factor
            )
            self.cost_cache[cache_key] = cost
            return cost
        
        # Extract state components
        day, fatigue, risk, performance, _ = state
        prev_day, prev_fatigue, prev_risk, prev_performance, _ = prev_state
        
        # Calculate deltas (changes in state)
        delta_fatigue = fatigue - prev_fatigue
        delta_risk = risk - prev_risk
        delta_performance = performance - prev_performance
        
        # Determine if this is a rest day
        is_rest = (action[0] == 0.0 and action[1] == 0.0)
        
        # Calculate efficiency metrics
        if is_rest:
            # Rest days should primarily focus on recovery
            # Lower cost for better recovery (more fatigue reduction)
            recovery_efficiency = max(0, prev_fatigue - fatigue)
            cost = 3.0 - 2.0 * recovery_efficiency
        else:
            # Training days should balance performance gain vs risk/fatigue increase
            
            # No performance improvement = higher cost
            if delta_performance <= 0:
                perf_factor = 5.0
            else:
                # Performance efficiency: lower cost for more performance gain per fatigue/risk
                fatigue_risk_sum = max(0.01, delta_fatigue + delta_risk * 5)
                perf_factor = 1.0 / (delta_performance / fatigue_risk_sum)
            
            # Risk penalty is more severe as we approach max risk
            risk_proximity = risk / self.max_risk if self.max_risk > 0 else 0
            risk_penalty = 2.0 * risk_proximity**2
            
            # Fatigue penalty is more severe as we approach max fatigue
            fatigue_proximity = fatigue / self.max_fatigue if self.max_fatigue > 0 else 0
            fatigue_penalty = 1.5 * fatigue_proximity**2
            
            # Combined cost (lower is better)
            cost = perf_factor + risk_penalty + fatigue_penalty
        
        self.cost_cache[cache_key] = cost
        return cost
    
    def heuristic(self, state):
        """
        Calculate a heuristic value estimating how promising a state is.
        
        This function estimates how good the current state is for reaching
        the goal state. It considers:
        1. Performance deficit and potential for improvement
        2. Risk level penalty
        3. Fatigue level penalty
        4. Distance from target day
        
        A lower heuristic value means the state is more promising.
        
        Args:
            state: The state to calculate the heuristic for
            
        Returns:
            A numerical heuristic value (lower is better)
        """
        cache_key = self.state_to_key(state)
        if cache_key in self.heuristic_cache:
            return self.heuristic_cache[cache_key]
            
        day, fatigue, risk, performance, _ = state
        
        # Calculate remaining days until target
        remaining_days = max(0, self.target_day - day)
        
        # Performance deficit from target
        perf_deficit = max(0, self.target_perf - performance)
        
        # Estimate future performance potential based on remaining days
        # Assume that performance can improve more with more remaining days
        performance_potential = perf_deficit * (1 + remaining_days * 0.1)
        
        # Risk penalty - higher risk states are less desirable
        risk_penalty = 2.0 * (risk / self.max_risk) if risk > 0.3 else 0
        
        # Fatigue penalty - higher fatigue states are less desirable
        fatigue_penalty = 1.5 * (fatigue / self.max_fatigue) if fatigue > 0.7 * self.max_fatigue else 0
        
        # Days penalty - ensure we prioritize reaching target day with good performance
        days_penalty = 0 if day >= self.target_day else 3.0 * (self.target_day - day) / self.target_day
        
        # Combined heuristic - lower is better
        h = performance_potential + risk_penalty + fatigue_penalty + days_penalty
        
        # Store in cache
        self.heuristic_cache[cache_key] = h
        return h

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
        """
        Initialize the greedy search algorithm with a problem instance.
        
        Args:
            problem: An AthletePerformanceProblem instance
        """
        self.problem = problem
        self.expanded_nodes = 0
        self.max_queue_size = 0
        
    def search(self, max_depth=float('inf')):
        """
        Perform greedy best-first search to find an optimal training plan.
        
        This function implements a greedy best-first search algorithm that uses
        a priority queue to always explore the most promising node first, based
        on the heuristic value. The search continues until a goal state is found
        or the entire search space is explored.
        
        Args:
            max_depth: Maximum search depth (default: infinity)
            
        Returns:
            The goal node if found, None otherwise
        """
        start_node = Node(state=self.problem.initial_state)
        
        # Use a priority queue to always explore the "best" node first
        # The item with the lowest priority value (heuristic) comes out first
        frontier = queue.PriorityQueue()
        frontier.put((0, start_node))  # (priority, node)
        
        # Track explored states to avoid cycles and repeated work
        explored = set()
        
        while not frontier.empty():
            # Get node with lowest priority value (heuristic)
            _, current_node = frontier.get()
            
            # If we've reached the target day, return this node
            if current_node.state[0] >= self.problem.target_day:
                return current_node
                
            # Skip already explored states (using rounded state values for efficiency)
            rounded_state = self._round_state(current_node.state)
            if rounded_state in explored:
                continue
            
            # Mark this state as explored
            explored.add(rounded_state)
            
            # Expand node - generate all possible children by applying actions
            children = self.problem.expand_node(
                current_node, 
                use_cost=True,
                use_heuristic=True
            )
            
            self.expanded_nodes += 1
            
            # For each child, add to frontier if not exceeding max depth
            for child in children:
                # Check if this child's day exceeds the max depth
                if child.depth > max_depth:
                    continue
                    
                # For greedy search, we primarily use the heuristic
                priority = child.h  # Use only heuristic for greedy best-first
                
                # Add to frontier
                frontier.put((priority, child))
                
            # Track maximum queue size for performance analysis
            self.max_queue_size = max(self.max_queue_size, frontier.qsize())
            
        # If we've examined all nodes and haven't found a solution, return None
        return None
        
    def _round_state(self, state):
        """
        Round state values to reduce the state space and avoid similar states.
        
        This function creates a simplified representation of a state by rounding
        continuous values to reduce the state space and avoid exploring nearly
        identical states.
        
        Args:
            state: The state to round
            
        Returns:
            A hashable tuple with rounded state values
        """
        day, fatigue, risk, performance, _ = state
        # Round values to reduce state space
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
        initial_state=(0, 1.5, 0.2, 6),  # Initial state: day 0, fatigue 1.5, risk 0.2, performance 6.0
        target_day=30,                    # Training plan duration: 30 days
        target_perf=7.0,                  # Target performance: 7.0 out of 10
        max_fatigue=3.5,                  # Maximum allowable fatigue: 3.5 out of 5
        max_risk=0.5                      # Maximum allowable risk: 0.5 out of 1
    )
    
    # Create the greedy search algorithm
    searcher = GreedySearch(problem)

    # Run the search algorithm
    goal_node = searcher.search()
    
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
            if action:  # Skip first None action
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
    test_greedy_search()
