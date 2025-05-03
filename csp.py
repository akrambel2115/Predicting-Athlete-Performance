import joblib
import numpy as np
import pandas as pd
import time
from Problem import AthletePerformanceProblem
from collections import defaultdict
import math

class AthleteTrainingCSP:
    """
    Implementation of a Constraint Satisfaction Problem approach for athlete training planning
    using only backtracking search.
    
    The CSP approach models the athlete training planning problem with:
    - Variables: Training activities for each day
    - Domains: Possible training intensities and durations for each day
    - Constraints: Fatigue limits and injury risk thresholds
    - Objective: Maximize performance
    """
    
    def __init__(self, 
        initial_state=(0, 0.0, 0.0, 1.0),
        target_day=30,
        max_fatigue=3.0,
        max_risk=0.5):
        
        """
        Initialize the CSP for athlete training planning.
        
        Args:
            initial_state: Tuple of (day, fatigue, risk, performance)
            target_day: Target training period length in days
            max_fatigue: Maximum allowable fatigue level
            max_risk: Maximum allowable injury risk
        """
        
        # an object of the problem definition
        self.athlete_problem = AthletePerformanceProblem(initial_state=initial_state)
        
        # Thresholds
        self.target_day = target_day
        self.max_fatigue = max_fatigue
        self.max_risk = max_risk
        
        # Initialize state with history
        day, f, r, p = initial_state
        self.initial_state = (day, f, r, p, [
            {'load': 0.0,
            'fatigue': f,
            'injury_count': 0,
            'days_since_game': 0,
            'days_since_last_injury': 0}
        ])
        
        self.intensities = [0.0, 0.3, 0.6, 0.9]
        self.durations = [0, 60, 90, 120]
        
        # the domains: possible intensities and durations
        self.domains = set()
        
        # all possible pairs of intensity and duration
        for intensity in self.intensities:
            for duration in self.durations:
                if (intensity==0.0 and duration==0) or (intensity> 0.0 and duration>0):
                    self.domains.add((intensity, duration))

        self.transition_cache = {} # Cache for performance optimization
        
        # backtracking stats
        self.backtrack_stats = {
            'iterations': 0,
            'max_depth': 0,
            'pruning_count': 0
        }

    def get_domains(self):
        return self.domains
    
    def apply_action(self, state, action):
        """
        Args:
            state: Tuple of (day, fatigue, risk, performance, history)
            action: Tuple of (intensity, duration)
            
        Returns:
            The resulting new state after applying the action
        """
        
        # caching to avoid recalculating
        cache_key = (str(state), str(action))
        if cache_key in self.transition_cache:
            return self.transition_cache[cache_key]
                
        day, fatigue, risk, performance, history = state
        
        basic_state = (day, fatigue, risk, performance, history) # tuple of only four elements !!!
        
        # apply the transition
        new_state = self.athlete_problem.apply_action(basic_state, action)
        
        self.transition_cache[cache_key] = new_state    # cache the answer
        return new_state
    
    def check_constraints(self, state):
        """Check if the current state satisfies all constraints"""
        _, fatigue, risk, _, _ = state
        return fatigue <= self.max_fatigue and risk <= self.max_risk

    def evaluate_solution(self, solution):
        """
        Evaluate a complete training plan.
        
        Args:
            solution: List of (intensity, duration) actions for each day
            
        Returns:
            Dictionary with evaluation metrics
        """
        current_state = self.initial_state
        states = [current_state]
        
        # Apply each action in the solution
        for action in solution:
            current_state = self.apply_action(current_state, action)
            states.append(current_state)
        
        # Get final state
        final_day, final_fatigue, final_risk, final_performance, _ = states[-1]
        
        # Calculate metrics
        highest_fatigue = max(state[1] for state in states)
        highest_risk = max(state[2] for state in states)
        constraints_violated = any(state[1] > self.max_fatigue or state[2] > self.max_risk for state in states)
        
        # Calculate training metrics
        rest_days = sum(1 for action in solution if action[0] == 0.0 and action[1] == 0)
        high_intensity_days = sum(1 for action in solution if action[0] >= 0.7)
        total_workload = sum(self.athlete_problem.LOAD_PER_MIN.get(action[0], 0) * action[1] for action in solution)
        
        return {
            'final_performance': final_performance,
            'highest_fatigue': highest_fatigue,
            'final_fatigue': final_fatigue,
            'final_risk': final_risk,
            'highest_risk': highest_risk,
            'constraints_violated': constraints_violated,
            'days_trained': final_day,
            'rest_days': rest_days,
            'high_intensity_days': high_intensity_days,
            'total_workload': total_workload
        }

    def backtracking_search(self, time_limit=120):
        """
        Optimized backtracking search algorithm to find the maximum performance solution.
        
        Args:
            time_limit: Maximum time in seconds to run the backtracking search
            
        Returns:
            The best solution found within the time limit
        """
        # Reset statistics
        self.backtrack_stats = {
            'iterations': 0,
            'max_depth': 0,
            'pruning_count': 0
        }
        
        best_solution = None
        best_performance = 0.0
        
        # backtracking (nested function)
        def _backtrack(assignment, current_state, depth=0):
            """
            Recursive backtracking function to find the solution with maximum performance.
            """
            nonlocal best_solution, best_performance
            
            # Update stats
            self.backtrack_stats['iterations'] += 1
            self.backtrack_stats['max_depth'] = max(self.backtrack_stats['max_depth'], depth)
            
            # check time (if it took more than the limit we set)
            if time.time() - start_time > time_limit:
                return
            
            day, fatigue, risk, performance, _ = current_state
            
            # Check if we've reached the target day
            if day >= self.target_day:
                # If this solution has better performance than our current best, update it
                if performance > best_performance:
                    best_performance = performance
                    best_solution = assignment.copy()
                return
            
            # Get possible actions for this day, ordered by most promising
            actions = self._get_ordered_domain_values(current_state)
            
            for action in actions:
                # Try this action
                new_state = self.apply_action(current_state, action)
                
                # Check if constraints are satisfied
                if self.check_constraints(new_state):
                    # Add action to assignment
                    assignment.append(action)
                    
                    # Recursively continue with updated state
                    _backtrack(assignment, new_state, depth + 1)
                    
                    # Backtrack - remove the action
                    assignment.pop()
        
        start_time = time.time()        
        _backtrack([], self.initial_state)
        
        print(f"Best performance found: {best_performance:.2f}")
        return best_solution
    
    def _get_ordered_domain_values(self, state):
        """
        Order domain values (actions) by most promising first for the current state.
        This function maximizes performance gain as the primary objective.
        
        Args:
            state: Current state which includes day, fatigue, risk, performance and history
            
        Returns:
            List of actions ordered by most promising first (best actions at the beginning)
        """
        # WEIGHTS TO define the priorities
        #########################
        PERFORMANCE_WEIGHT= 100
        POTENTIAL_WEIGHT= 15
        LOW_FATIGUE_WEIGHT= -5
        LOW_RISK_WEIGHT= -5
        TRAINING_EFFICIENCY= 5
        ########################
        
        
        day, fatigue, risk, performance, _ = state
        
        actions = self.get_domains()
        
        # store the calculated action:priority 
        action_values = {}
        
        for action in actions:
            intensity, duration = action
            
            future_state = self.apply_action(state, action)
            
            _, future_fatigue, future_risk, future_performance, _ = future_state
            
            # we will use this to assign each action a priority
            perf_improvement = future_performance - performance
            risk_added = future_risk - risk
            fatigue_added= future_fatigue - fatigue 
            # long_term_potential = intensity * (duration / 60) * 0.05  # estimated future gains

            if future_fatigue >= self.max_fatigue or future_risk >= self.max_risk:
                # very negative value to actions that violate constraints
                action_values[action] = -float('inf')
                            
            else:      
                # performance improvement per hour
                training_efficiency = perf_improvement / max(0.5, duration/60)
                
                ### with this formula we give the max priority to the performance 
                ### and also taking in consideration some other things
                action_values[action] = (
                    perf_improvement * PERFORMANCE_WEIGHT +
                    training_efficiency * TRAINING_EFFICIENCY +
                    risk_added * LOW_RISK_WEIGHT +       # weights of risk and fatigue are negative
                    fatigue_added * LOW_FATIGUE_WEIGHT
                )
                
                # we can add some other bonuses for some things ..
                # TODO
                
                
        # Sort actions by their calculated values in descending order
        return sorted(actions, key=lambda a: action_values.get(a, 0), reverse=True)

def test_backtracking_csp_max_performance():
    """Test the CSP approach with optimized backtracking for maximum performance."""
    
    print("Testing CSP Athlete Training Planner for Maximum Performance")
    print("---------------------------------------------------------")
    
    # CSP problem with specific parameters
    problem = AthleteTrainingCSP(
        initial_state=(0, 1.5, 0.1, 6.0),  # initial state
        target_day=30,                    
        max_fatigue=3,                 
        max_risk=0.22                      
    )
    
    print("Finding solution to maximize performance...")
    start_time = time.time()
    solution = problem.backtracking_search(time_limit=120)  # 2 minute time limit
    end_time = time.time()
    
    print(f"Search completed in {end_time - start_time:.2f} seconds")
    print(f"Backtracking iterations: {problem.backtrack_stats['iterations']}")
    print(f"Maximum depth reached: {problem.backtrack_stats['max_depth']}")
    print(f"Number of branches pruned: {problem.backtrack_stats['pruning_count']}")
    
    if solution is None:
        print("No solution found.")
        return
    
    # Display the training plan
    print("\nTraining Plan:")
    print("Day | Intensity | Duration | Fatigue | Risk | Performance")
    print("----|-----------|----------|---------|------|------------")
    
    # Display initial state
    state = problem.initial_state
    day = 0
    print(f"{day:3d} | {'-':9} | {'-':8} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")
    
    # Display each day in the training plan
    for i, action in enumerate(solution):
        state = problem.apply_action(state, action)
        day = i + 1
        intensity, duration = action
        print(f"{day:3d} | {intensity:9.1f} | {duration:8.1f} |  {state[1]:.2f}   | {state[2]:.2f} | {state[3]:.2f}")
    
    # Display final state summary
    final_day, final_fatigue, final_risk, final_perf, _ = state
    print("\nFinal State:")
    print(f"Day: {final_day}")
    print(f"Fatigue: {final_fatigue:.2f}/{problem.max_fatigue:.2f}")
    print(f"Risk: {final_risk:.2f}/{problem.max_risk:.2f}")
    print(f"Performance: {final_perf:.2f}/10.00")
    
    # Evaluate solution
    evaluation = problem.evaluate_solution(solution)
    print("\nSolution Evaluation:")
    print(f"Final Performance: {evaluation['final_performance']:.2f}")
    print(f"Constraints Violated: {'Yes' if evaluation['constraints_violated'] else 'No'}")
    print(f"Highest Fatigue: {evaluation['highest_fatigue']:.2f}/{problem.max_fatigue:.2f}")
    print(f"Highest Risk: {evaluation['highest_risk']:.2f}/{problem.max_risk:.2f}")
    print(f"Rest Days: {evaluation['rest_days']}/{evaluation['days_trained']}")
    print(f"High Intensity Days: {evaluation['high_intensity_days']}")
    print(f"Total Workload: {evaluation['total_workload']:.2f}")

if __name__ == "__main__":
    test_backtracking_csp_max_performance()