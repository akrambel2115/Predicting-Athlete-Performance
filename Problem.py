import joblib
import numpy as np
import pandas as pd
from calculate_load_per_minute import calculate_load_per_minute
import Node
import random

class AthletePerformanceProblem:
    """
    Search problem for athlete performance planning using learned ΔF, ΔP, ΔR models.
    State: (day, fatigue, risk, performance, history)
    Actions: Train (intensity, duration) or Rest: (0.0, 0.0).
    Transition: ML regression/classification models via simulate_step logic.
    Cost: customizable weighted sum (not implemented here).
    """
    def __init__(self,
                 initial_state: tuple = (0, 1.0, 0.1, 5.0),
                 target_day: int = 10,
                 genetic: bool = False):
        # Load models
        if genetic:
            self.delta_f = joblib.load("genetic_model/delta_f_model.pkl")
            self.delta_p = joblib.load("genetic_model/delta_p_model.pkl")
            r_loaded = joblib.load("predictingModels/delta_r_classifier.pkl")
        else:
            self.delta_f = joblib.load("predictingModels/delta_f_model.pkl")
            self.delta_p = joblib.load("predictingModels/delta_p_model.pkl")
            r_loaded = joblib.load("predictingModels/delta_r_classifier.pkl")

        # Unpack classifier
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
        # Compute load-per-minute mapping
        self.LOAD_PER_MIN = calculate_load_per_minute()
        # Defaults
        self.SLEEP_DUR = 7.5
        self.SLEEP_QLT = 3.0
        self.STRESS    = 2.5
        self.f_feats = list(self.delta_f.feature_names_in_)
        self.p_feats = list(self.delta_p.feature_names_in_)
        # Initialize state history
        day, f, r, p = initial_state
        self.initial_state = (day, f, r, p, [
            {'load': 0.0,
             'fatigue': f,
             'injury_count': 0,
             'days_since_game': 0,
             'days_since_last_injury': 0}
        ])

        self.target_day = target_day

    def actions(self):
        train_actions = [(i, d) for i in (0.3, 0.6, 0.9) for d in (60, 120)]
        return train_actions + [(0.0, 0.0)]  # rest

    def apply_action(self, state, action):
        # Unpack
        day, F, R, P, history = state
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
        
        if is_rest:
            Rn = np.clip(R * 0.86, 0.0, 1.0)
            Fn = max(F * 0.85, 0.0)
            Pn = max(P * 0.91, 0.0)
        else:
            dF = float(self.delta_f.predict(X[self.f_feats])[0])
            dP = float(self.delta_p.predict(X[self.p_feats])[0])
            prob = self.delta_r.predict_proba(X[self.r_feats])[0, 1]
            Rn = np.clip(R + prob, 0.0, 1.0)
            Fn = np.clip(F + dF, 0.0, 5.0)
            Pn = np.clip(P + dP, 0.0, 10.0)

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
    
    def expand_node(self, node,use_cost=False, use_heuristic=False):
        """
        Expands a node by applying all possible actions and returning the resulting nodes.
        """
        children = []
        for action in self.actions():
            new_state = self.apply_action(node.state, action)
            if self.is_valid(new_state):
                cost = self.cost(node.state, action) if use_cost else 0
                heuristic = self.heuristic(new_state) if use_heuristic else 0
                child_node = Node(new_state, parent=node, action=action, cost=cost, f=cost + heuristic)
                children.append(child_node)
        return children

    def is_valid(self, state):
        day, fatigue, risk, performance, _ = state
        return (
                0 <= fatigue <= 5.0
                and 0 <= risk <= 1.0
                and 0 <= performance <= 10.0) # mata5rbouch fjadhoum

    def is_goal(self, state):
        day, fatigue, risk, performance, _ = state
        return (day == self.target_day
                and performance >= self.target_perf
                and fatigue <= self.max_fatigue
                and risk <= self.max_risk)
    
    def cost(self, state, action):
        """
        Calculate the cost of applying an action to the current state.
        
        This function evaluates the immediate cost of applying an action, balancing:
        - Performance improvement (negative cost/benefit)
        - Increase in fatigue (cost)
        - Increase in injury risk (cost)
        - Training load (cost proportional to intensity×duration)
        
        Lower cost values indicate better actions.
        
        Args:
            state: Current state (day, fatigue, risk, performance, history)
            action: Action to apply (intensity, duration)
            
        Returns:
            Numerical cost value (lower is better)
        """
        day, fatigue, risk, performance, _ = state
        intensity, duration = action
        is_rest = (intensity == 0.0 and duration == 0.0)
        
        # Apply action to get new state
        new_state = self.apply_action(state, action)
        _, new_fatigue, new_risk, new_perf, _ = new_state
        
        # Calculate deltas (changes in state)
        delta_fatigue = new_fatigue - fatigue
        delta_risk = new_risk - risk
        delta_perf = new_perf - performance
        
        if is_rest:
            # For rest days, prioritize recovery (fatigue reduction)
            recovery_efficiency = max(0, fatigue - new_fatigue)
            cost = 2.0 - (3.0 * recovery_efficiency)
            # Small penalty if performance drops significantly during rest
            if delta_perf < -1.0:
                cost += 1.0
        else:
            # For training days, calculate efficiency metrics
            # Higher cost if no performance improvement
            if delta_perf <= 0:
                perf_factor = 5.0
            else:
                # Performance efficiency: lower cost for more performance gain relative to fatigue/risk
                # Add small constant to avoid division by zero
                fatigue_risk_sum = max(0.01, delta_fatigue + (delta_risk * 4.0))
                perf_factor = 2.0 - min(2.0, delta_perf / fatigue_risk_sum)
            
            # Risk penalty increases exponentially as we approach maximum risk
            risk_proximity = new_risk
            risk_penalty = 2.0 * (risk_proximity ** 2)
            
            # Fatigue penalty increases as we approach maximum fatigue (assumed to be 5.0)
            fatigue_proximity = new_fatigue / 5.0
            fatigue_penalty = 1.5 * (fatigue_proximity ** 2)
            
            # Combined cost (lower is better)
            cost = perf_factor + risk_penalty + fatigue_penalty
            
            # Add penalty for excessive training load
            training_load = intensity * duration
            if training_load > 80:
                cost += 0.5 * (training_load - 80) / 20
        
        return cost

    def heuristic(self, state) -> float:
        """
        Estimate how close the current state is to the goal state.
        
        This function provides a heuristic that considers:
        1. Performance deficit from target
        2. Days remaining to reach target
        3. Current fatigue and risk levels
        4. Potential for improvement over remaining days
        
        Lower heuristic values indicate more promising states.
        
        Args:
            state: Current state (day, fatigue, risk, performance, history)
            
        Returns:
            Numerical heuristic value (lower is better)
        """
        day, fatigue, risk, performance, _ = state
        
        # Calculate remaining days until target
        remaining_days = max(0, self.target_day - day)
        
        # Calculate performance deficit from target
        perf_deficit = max(0, self.target_perf - performance)
                
        # If already at the target day, evaluate based on goal conditions
        if remaining_days == 0:
            # If performance goal met and constraints satisfied, heuristic is 0
            if (performance >= self.target_perf and 
                fatigue <= self.max_fatigue and 
                risk <= self.max_risk):
                return 0.0
            
            # Otherwise, return a value based on how far we are from satisfying all conditions
            return (
                3.0 * perf_deficit + 
                2.0 * max(0, fatigue - self.max_fatigue) +
                2.0 * max(0, risk - self.max_risk)
            )
        
        # For states before the target day, estimate based on trajectory
        
        # Estimate max potential performance improvement per day, simplified and should be studied from the transition model
        max_improvement_per_day = 0.3
        
        # Estimate if we can reach the performance target in time
        potential_improvement = max_improvement_per_day * remaining_days
        if potential_improvement < perf_deficit:
            # Cannot reach target with max improvement rate, so increase heuristic
            reachability_penalty = 2.0 * (perf_deficit - potential_improvement)
        else:
            reachability_penalty = 0.0
        
        # Risk and fatigue penalties increase as we get closer to max allowed values
        risk_proximity = risk / self.max_risk if hasattr(self, 'max_risk') else risk
        fatigue_proximity = fatigue / self.max_fatigue if hasattr(self, 'max_fatigue') else fatigue / 5.0
        
        risk_penalty = 1.5 * risk_proximity**2
        fatigue_penalty = 1.0 * fatigue_proximity**2
        
        # Days factor - prioritize states that have made more progress toward goal
        days_factor = 0.8 * (1.0 - day / self.target_day)
        
        # Combined heuristic - lower values are better
        return perf_deficit + reachability_penalty + risk_penalty + fatigue_penalty + days_factor

    def random_individual(self):
        """
        Create a random training schedule for the target number of days.
        
        Returns:
            A tuple of (intensity, duration) pairs representing a training schedule
        """
        # Default to 14 days if target_day is not set
        days = getattr(self, 'target_day', 14)
        
        # Possible intensities and durations
        intensities = [0.0, 0.3, 0.6, 0.9]  # Including rest days (0.0)
        durations = [0, 30, 60, 90, 120]    # 0 for rest days
        
        # Generate random schedule
        schedule = []
        for _ in range(days):
            intensity = random.choice(intensities)
            # If it's a rest day, duration is 0
            duration = 0 if intensity == 0.0 else random.choice(durations[1:])
            schedule.append((intensity, duration))
        return list(schedule)

    def evaluate_individual(self, indiv):
        
        current_state = self.initial_state
        individual = indiv[:]
        while individual:
            indiv_action = individual.pop(0)
            current_state = self.apply_action(current_state, indiv_action) 

        return current_state[:-1]
