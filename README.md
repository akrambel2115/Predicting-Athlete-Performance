# Predicting Athlete Performance: A Data-Driven Approach to Training Optimization

![Project Banner](https://your-image-url-here.com/banner.png) <!-- Optional: Add a project banner image -->

## 📖 Table of Contents

- [📍 Introduction](#-introduction)
- [✨ Key Features](#-key-features)
- [🤖 Algorithms & Models](#-algorithms--models)
- [🛠️ Tech Stack](#-tech-stack)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [🏃‍♂️ Usage](#-usage)
- [📄 License](#-license)
## 📍 Introduction

**Predicting Athlete Performance** is an intelligent system designed to create optimal training plans for athletes. By leveraging AI search algorithms and machine learning models, this project analyzes historical performance data to generate personalized training schedules that maximize performance while minimizing fatigue and injury risk.

The core of the project is a web application that provides a user-friendly interface for athletes and coaches to input parameters and receive a tailored training plan. The system models the training process as a state-space search problem, where each state represents the athlete's condition (fatigue, injury risk, performance) and actions represent daily training activities (intensity and duration).

## ✨ Key Features

- **Personalized Training Plans:** Generates multi-day training schedules based on individual athlete data.
- **Performance Optimization:** Aims to maximize athlete performance over a given period.
- **Injury and Fatigue Management:** Adheres to configurable constraints for maximum fatigue and injury risk.
- **Multiple AI Algorithms:** Implements a variety of search algorithms for comparison and optimal plan generation, including:
  - Uninformed Search: BFS, DFS, UCS
  - Informed Search: A*, Greedy Search
  - Advanced Methods: Constraint Satisfaction (CSP), Genetic Algorithms
- **Web-Based Interface:** An intuitive Flask-based web application for easy interaction and visualization of results.
- **Data-Driven Models:** Uses machine learning models trained on soccer data to predict fatigue, injury risk, and performance.

## 🤖 Algorithms & Models

This project explores and implements several AI techniques to solve the training optimization problem:

| Algorithm/Model             | File(s)                                      |
| --------------------------- | -------------------------------------------- |
| **Uninformed Search**       | `BFS_search.py`, `DFS_search.py`, `UCS_Search.py` |
| **Informed Search**         | `A_star.py`, `greedy_search_implementation.py` |
| **Constraint Satisfaction** | `csp.py`                                     |
| **Genetic Algorithm**       | `Genetic.py`, `Genetic_akram.py`             |
| **Prediction Models**       | `predictingModels/`, `genetic_model/`        |

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Core Libraries:** Pandas, Scikit-learn, NumPy
- **Visualization:** Chart.jsm, Matplotlib

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Make sure you have Python and pip installed.

- **Python** (version 3.8 or higher recommended)
- **pip** (Python package installer)

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/akrambel2115/Predicting-Athlete-Performance.git
   cd Predicting-Athlete-Performance
   ```

2. **Create and activate a virtual environment (recommended):**
   ```sh
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

1. **Navigate to the website directory:**
   ```sh
   cd Predicting-Athlete-Performance-Website
   ```

2. **Run the Flask application:**
   ```sh
   python app.py
   ```

3. **Open your web browser** and go to `http://127.0.0.1:5000` to access the application.

From the web interface, you can:
- Set initial athlete parameters (fatigue, risk, performance).
- Define the training period and constraints.
- Select an AI algorithm to generate a training plan.
- View the optimized training schedule and performance projections.


## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
Project Link: [https://github.com/akrambel2115/Predicting-Athlete-Performance](https://github.com/akrambel2115/Predicting-Athlete-Performance)
