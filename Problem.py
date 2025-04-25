import joblib
import numpy as np
import pandas as pd
from calculate_load_per_minute import calculate_load_per_minute

class AthletePerformanceProblem:
    """
    Search problem for athlete performance planning using learned ΔF, ΔP, ΔR models.
    State: (day, fatigue, risk, performance, history)
    Actions: Train (intensity, duration) or Rest: (0.0, 0.0).
    Transition: ML regression/classification models via simulate_step logic.
    Cost: customizable weighted sum (not implemented here).
    """
    def __init__(self,
                 initial_state: tuple = (0, 0.0, 0.0, 50.0),
                 w1: float = 1.0,
                 w2: float = 1.0,
                 w3: float = 1.0):
        # Load models
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
        self.w1, self.w2, self.w3 = w1, w2, w3
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

    def actions(self):
        train_actions = [(i, d) for i in (0.3, 0.6, 0.9) for d in (60, 90, 120)]
        return train_actions + [(0.0, 0.0)]  # rest

    def apply_action(self, state, action, history):
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
            F_new = max(F * 0.85, 0.0)
            P_new = max(P * 0.96, 0.0)
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
    
    def cost(self, state, action):
        w = self.weights
        _, fatigue, risk, performance = state
        _, fatigue_p, risk_p, performance_p = self.apply_action(state, action)
        delta_f, delta_r, delta_p = fatigue_p - fatigue, risk_p - risk, performance_p - performance

        # Calculate the cost as a weighted sum of performance deficit, risk, and fatigue
        return (w['w1'] * delta_f + w['w2'] * delta_r - w['w3'] * delta_p + w['w4'] * action[0] * action[1])
    
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
        day, fatigue, risk, performance = state
        return (
                0 <= fatigue <= 1.0
                and 0 <= risk <= 1.0
                and 0 <= performance <= 100.0)

    def is_goal(self, state):
        return (state.day == self.target_day
                and state.performance >= self.target_perf
                and state.fatigue <= self.max_fatigue
                and state.risk <= self.max_risk)
    
    
    def heuristic(self, state) -> float:
        max_I, max_D = 0.9, 90
        eta = self.coeffs['eta']
        remaining = max(0.0, self.target_perf - state.performance)
        best_gain_per_day = eta * max_I * max_D
        return remaining / best_gain_per_day if best_gain_per_day > 0 else 0.0
