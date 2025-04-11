# Dataset Overview

This dataset is designed to analyze the impact of complex scheduling algorithms on injury rates and athletic performance in a collegiate sports environment. It provides synthetic but realistic data for athletes, capturing their demographics, training regimes, schedules, fatigue levels, and injury risks.

## Features Overview

### 1. Athlete Information
- **Athlete_ID**: Unique identifier for each athlete (e.g., A001, A002).
- **Age**: Athlete's age (18–25 years).
- **Gender**: Gender of the athlete (Male/Female).
- **Height_cm**: Height of the athlete in centimeters (160–200 cm).
- **Weight_kg**: Weight of the athlete in kilograms (55–100 kg).
- **Position**: Playing position in the team (Guard, Forward, Center).

### 2. Training Information
- **Training_Intensity**: Average intensity of training sessions on a scale of 1 (low) to 10 (high).
- **Training_Hours_Per_Week**: Total hours of training per week (5–20 hours).
- **Recovery_Days_Per_Week**: Number of days dedicated to recovery per week (1–3 days).

### 3. Schedule Information
- **Match_Count_Per_Week**: Number of matches scheduled per week (1–4 matches).
- **Rest_Between_Events_Days**: Average rest days between matches (1–3 days).

### 4. Derived Features
- **Load_Balance_Score**: A calculated score (0–100) indicating the balance between training load and recovery. A higher score reflects a better balance.
- **ACL_Risk_Score**: Predicted risk score (0–100) for ACL injuries. A higher score indicates a greater risk of injury.

### 5. Injury Information
- **Injury_Indicator**: Target column indicating whether the athlete sustained an ACL injury (1 = Yes, 0 = No).

### 6. Performance Metrics
- **Fatigue_Score**: Subjective fatigue level on a scale of 1 (low) to 10 (high).
- **Performance_Score**: Composite performance score (50–100) based on metrics like points scored and assists.
- **Team_Contribution_Score**: Athlete’s overall contribution to the team’s success on a scale of 50–100.

---
*This dataset is intended for use in analyzing the correlation between scheduling, training loads, recovery, and injury risk in collegiate athletes.*
