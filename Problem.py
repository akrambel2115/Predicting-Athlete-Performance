class AthletePerformanceProblem:
    """
    Defines a search problem for athlete performance planning.
    State: (day, fatigue, risk, performance)
    Actions: Train: (intensity_value, duration) or Rest: (0,0)
    Transition: deterministic model with tunable coefficients
    Cost: weighted sum of performance deficit, risk, and fatigue
    """
    def __init__(self,
                 initial_state: tuple = (0, 0.0, 0.0, 100.0),
                 alpha: float = 0.8,
                 beta: float = 0.001,
                 gamma: float = 0.1,
                 delta: float = 0.05,
                 epsilon: float = 0.001,
                 eta: float = 0.002,
                 theta: float = 0.1,
                 w1: float = 1.0,
                 w2: float = 1.0,
                 w3: float = 1.0,
                 w4: float = 1.0,
                 target_day: int = 30,
                 target_perf: float = 95.0,
                 max_fatigue: float = 0.5,
                 max_risk: float = 0.3):
        self.coeffs = dict(alpha=alpha, beta=beta, gamma=gamma,
                           delta=delta, epsilon=epsilon,
                           eta=eta, theta=theta)
        self.weights = dict(w1=w1, w2=w2, w3=w3)
        self.target_day = target_day
        self.target_perf = target_perf
        self.max_fatigue = max_fatigue
        self.max_risk = max_risk
        self.initial_state = initial_state

    def actions(self):
        train_actions = [(intensity, duration)
                        for intensity in (0.3, 0.6, 0.9)
                        for duration in (30, 60, 90)]
        return [train_actions, (0, 0)]  # Rest action

    def apply_action(self, state, action):
        day, fatigue, risk, performance = state
        intensity, duration = action
        coeffs = self.coeffs

        # Calculate new fatigue (F')
        new_fatigue = coeffs['alpha'] * fatigue + coeffs['beta'] * (intensity * duration) - coeffs['gamma'] * (1 - intensity)

        # Calculate new risk (R')
        new_risk = risk + coeffs['delta'] * new_fatigue + coeffs['epsilon'] * (intensity * duration)

        # Calculate new performance (P')
        new_performance = performance + coeffs['eta'] * (intensity * duration) - coeffs['theta'] * new_fatigue

        # Return the updated state
        return (day + 1, new_fatigue, new_risk, new_performance)
    
    def cost(self, state, action):
        w = self.weights
        fatigue, risk, performance = state
        fatigue_p, risk_p, performance_p = self.apply_action(state, action)
        delta_f, delta_r, delta_p = fatigue_p - fatigue, risk_p - risk, performance_p - performance

        # Calculate the cost as a weighted sum of performance deficit, risk, and fatigue
        return (w['w1'] * delta_f + w['w2'] * delta_r - w['w3'] * delta_p + w['w4'] * action[0] * action[1])


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
