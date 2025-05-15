
from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
import os
from datetime import datetime
import json
import secrets  # for better secret key generation
import sys
import importlib.util
import re

# Import search algorithms from the functions directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))
from functions.Problem import AthletePerformanceProblem
from functions.A_star import AStarSearch
from functions.greedy_search_implementation import GreedySearch
from functions.BFS_search import BFSSearch
from functions.dfs_search import DFSSearch
from functions.UCS_Search import UCSSearch
from functions.Genetic import GeneticAlgorithm
from functions.csp import AthleteTrainingCSP
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # More secure secret key

# In-memory storage for users (replace with database in production)
users = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # For demo purposes: allow any login if no users exist
        if not users:
            session['logged_in'] = True
            session['username'] = username
            flash('Welcome to the demo!', 'success')
            return redirect(url_for('dashboard'))
            
        if username in users and users[username]['password'] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('Successfully logged in!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    
    if not all([name, email, password]):
        flash('All fields are required.', 'error')
        return redirect(url_for('login'))
        
    if email in users:
        flash('Email already exists. Please use a different email or login.', 'error')
        return redirect(url_for('login'))
        
    users[email] = {
        'name': name,
        'password': password
    }
    session['logged_in'] = True
    session['username'] = email
    flash('Account created successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    
    current_date = datetime.now().strftime("%b %d, %Y")
    return render_template('dashboard.html', active_page='overview', current_date=current_date)

@app.route('/schedule')
def schedule():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    current_date = datetime.now().strftime("%b %d, %Y")
    return render_template('dashboard.html', active_page='schedule', current_date=current_date)

@app.route('/predictions')
def predictions():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    current_date = datetime.now().strftime("%b %d, %Y")
    return render_template('dashboard.html', active_page='predictions', current_date=current_date)

@app.route('/history')
def history():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    current_date = datetime.now().strftime("%b %d, %Y")
    return render_template('dashboard.html', active_page='history', current_date=current_date)

@app.route('/settings')
def settings():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    current_date = datetime.now().strftime("%b %d, %Y")
    return render_template('dashboard.html', active_page='settings', current_date=current_date)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    current_date = datetime.now().strftime("%b %d, %Y")
    return render_template('search.html', current_date=current_date)

@app.route('/api/run_search', methods=['POST'])

def run_search():
    algorithm_map = {
        'ucs': UCSSearch,
        'greedy': GreedySearch,
        'bfs': BFSSearch,
        'dfs': DFSSearch,
        'astar': AStarSearch,
        'genetic': GeneticAlgorithm,
        'csp': AthleteTrainingCSP

    }
    data = request.get_json()
    init = data.get('initialState', {})
    goal = data.get('goalState', {})
    algo_key = data.get('algorithm', 'astar')
    params = data.get('advancedParams', {})
    print(data)
    print(init)
    print(goal)
    print(algo_key)
    

    # Convert camelCase params to snake_case
    snake_case_params = {}
    for key, value in params.items():
        snake_key = re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()
        snake_case_params[snake_key] = value

    # Create Problem instance with initial and goal states
    try:
        initial_state = (
            0,  # Starting day
            init['fatigue'],
            init['risk'],
            init['performance']
        )
        if algo_key!='csp':
            problem = AthletePerformanceProblem(
                initial_state=initial_state,
                target_day=goal['days'],
                target_perf=goal['targetPerformance'],
                target_fatigue=goal['maxFatigue'],
                target_risk=goal['maxRisk'],
                genetic= (algo_key=='genetic')
            )
        else:
            initial_state = (
                0,
                init['fatigue'],
                init['risk'],
                init['performance']
            )
            problem = AthleteTrainingCSP(
                initial_state=initial_state,
                target_day=goal['days'],
                target_fatigue=goal['maxFatigue'],
                target_risk=goal['maxRisk']
            )
            
    except KeyError as e:
        return jsonify(success=False, error=f"Missing parameter: {e}"), 400

    # Get algorithm class
    AlgoClass = algorithm_map.get(algo_key)
    if not AlgoClass:
        return jsonify(success=False, error=f"Unknown algorithm: {algo_key}"), 400

    try:
        # Initialize algorithm with problem
        if algo_key=='csp':
            result = problem.search()
            return jsonify(result)            

        elif algo_key=='genetic':
            algorithm = AlgoClass(problem)
            result = algorithm.run()
            return jsonify(result)
        else:
            algorithm = AlgoClass(problem)
            result = algorithm.search()
            return jsonify(result)
            
        # Execute search with parameters

    except Exception as e:
        app.logger.exception("Search failed")
        return jsonify(success=False, error=str(e)), 500
    
# API endpoint to get the current search progress
@app.route('/api/search_progress', methods=['GET'])
def search_progress():
    # In a real application, this would retrieve the current progress from an ongoing search
    # For demonstration, we'll just return mock progress data
    mock_progress = {
        'nodesExplored': 756,
        'queueSize': 124,
        'elapsedTime': 1.5
    }
    return jsonify(mock_progress)

if __name__ == '__main__':
    app.run(debug=True)