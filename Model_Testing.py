"""
Simple module: curated test plans with descriptions
"""

# Each entry maps a label to a dict with:
# - "plan": list of (intensity, duration) tuples
# - "description": brief purpose & expected outcome

test_plans = {
    "all_rest": {
        "plan": [(0, 0)] * 14,
        "description": (
            "No training stimulus; all training actions are zero, "
            "fitness_gain=0, fatigue=0"
        )
    },
    "all_high_intensity_long": {
        "plan": [(0.9, 120)] * 14,
        "description": (
            "Extreme overload;"
            "expect overtraining flag"
        )
    },
    "mixed_full_length": {
        "plan": [
            (0.3, 60),  (0.6, 90),  (0.9, 120),
            (0.3, 90),  (0.6, 120), (0.9, 60),
            (0.3, 120), (0.6, 60),  (0.9, 90),
            (0.3, 60),  (0.6, 90),  (0.9, 120),
            (0.3, 90),  (0.6, 120)
        ],
        "description": (
            "Full-length mix of every intensity×duration; "
        )
    },
    "all_high_intensity_short": {
        "plan": [(0.9, 60)] * 10,
        "description": (
            "High-intensity short sessions; total_volume=540, "
            "expect high-stress but manageable fatigue"
        )
    },
    "alternating_rest_high": {
        "plan": [(0.9, 90) if i % 2 == 0 else (0, 0) for i in range(14)],
        "description": (
            "Alternating rest & high-intensity; total_volume=567, "
            "test fatigue/recovery logic"
        )
    },
    "empty_plan": {
        "plan": [],
        "description": (
            "Empty plan edge-case; total_volume=0, "
            "expect ""empty plan"" message"
        )
    },
    "single_moderate_session": {
        "plan": [(0.6, 90)],
        "description": (
            "Single moderate session; total_volume=54, "
            "expect low-load feedback"
        )
    },
    "exactly_24h_total": {
        "plan": [(0.6, 120)] * 12,
        "description": (
            "Total 1440min (~24h); expect ""exceeds limit"" flag"
        )
    },
    "many_low_intensity": {
        "plan": [(0.3, 30)] * 14,
        "description": (
            "Low-intensity high-frequency; total_volume=252, "
            "expect insufficient-overload warning"
        )
    }
}


if __name__ == "__main__":
    from Problem import AthletePerformanceProblem
    
    problem = AthletePerformanceProblem(genetic=True)
    for label, info in test_plans.items():
        plan = info["plan"]
        description = info["description"]
        print(f"\n=== Test Plan: {label} ===")
        print(f"Description: {description}")
        
        # Initialize the state with the problem's initial state
        state = problem.initial_state
        day, fatigue, risk, performance, _ = state
        
        print(f"Initial State - Day: {day}, Fatigue: {fatigue:.2f}, Risk: {risk:.2f}, Performance: {performance:.2f}")
        print(f"\nDay | Intensity | Duration | Fatigue | Risk | Performance")
        print(f"----|-----------|----------|---------|------|------------")
        
        # Apply each action in the plan and print the daily results
        for i, action in enumerate(plan):
            intensity, duration = action
            state = problem.apply_action(state, action)
            day, fatigue, risk, performance, _ = state
            
            print(f"{day:3d} | {intensity:9.1f} | {duration:8.0f} | {fatigue:7.2f} | {risk:4.2f} | {performance:10.2f}")
        
        # Print the final result summary
        final_day, final_fatigue, final_risk, final_performance, _ = state
        print(f"\nFinal Result: \n\t Days of Training: {final_day} \n\t Fatigue at Last Day: {final_fatigue:.2f} \n\t Risk of Injury: {final_risk:.2f} \n\t Performance: {final_performance:.2f}")

