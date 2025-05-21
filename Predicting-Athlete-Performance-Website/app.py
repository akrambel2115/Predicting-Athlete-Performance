from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
import os
from datetime import datetime, timedelta
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
app.permanent_session_lifetime = timedelta(days=7)  # Sessions last 7 days

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
            session.permanent = True  # Make session cookie persistent
            session['logged_in'] = True
            session['username'] = username
            flash('Welcome to the demo!', 'success')
            return redirect(url_for('dashboard'))
            
        if username in users and users[username]['password'] == password:
            session.permanent = True  # Make session cookie persistent
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
    session.permanent = True  # Make session cookie persistent
    session['logged_in'] = True
    session['username'] = email
    flash('Account created successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()  # This will remove the session cookie
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
    physic = data.get('physicological_parameters', {})
    goal = data.get('goalState', {})
    algo_key = data.get('algorithm', 'astar')
    params = data.get('advancedParams', {})

    print(physic)
    
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
                sleep_duration=physic['sleepDuration'],
                sleep_quality=physic['sleepQuality'],
                stress_level=physic['stressLevel']
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
                target_risk=goal['maxRisk'],
                sleep_duration=physic['sleepDuration'],
                sleep_quality=physic['sleepQuality'],
                stress_level=physic['stressLevel']
            )
            print(problem)
            
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
        elif algo_key=='genetic':
            algorithm = AlgoClass(problem, population_size= params['populationSize'], num_generations=params['generations'], mutation_rate= params['mutationRate'])
            result = algorithm.run()
        else:
            algorithm = AlgoClass(problem)
            result = algorithm.search()

        # Extract daily metrics from the result
        daily_metrics = []
        if isinstance(result, dict) and 'path' in result:
            # Extract metrics from each state in the path
            for state in result['path']:
                if isinstance(state, tuple) and len(state) >= 4:
                    day, fatigue, risk, performance = state
                    # Get intensity from the action if available
                    intensity = 0.3  # default value
                    if 'action' in result and isinstance(result['action'], dict):
                        intensity = result['action'].get('intensity', 0.3)
                    
                    daily_metrics.append({
                        'day': day,
                        'fatigue': fatigue,
                        'risk': risk,
                        'performance': performance,
                        'intensity': intensity
                    })

        # Prepare the data to save
        search_data = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': algo_key,
            'initial_state': init,
            'physiological_parameters': physic,
            'goal_state': goal,
            'parameters': params,
            'result': result,
            'daily_metrics': daily_metrics
        }

        # Save to current folder
        current_dir = os.path.join(os.path.dirname(__file__), 'static', 'js', 'current')
        os.makedirs(current_dir, exist_ok=True)
        with open(os.path.join(current_dir, 'current_search.json'), 'w') as f:
            json.dump(search_data, f, indent=4)

        # Save to history folder with timestamp
        history_dir = os.path.join(os.path.dirname(__file__), 'static', 'js', 'history')
        os.makedirs(history_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = os.path.join(history_dir, f'search_{timestamp}.json')
        with open(history_file, 'w') as f:
            json.dump(search_data, f, indent=4)

        return jsonify(result)
            
    except Exception as e:
        app.logger.exception("Search failed")
        return jsonify(success=False, error=str(e)), 500
    
# API endpoint to get the current search progress
@app.route('/api/search_progress', methods=['GET'])
def search_progress():
    # In a real application, this would retrieve the current progress from an ongoing search
    # For demonstration, we'll just return mock progress data
    return jsonify({
        'progress': 75,
        'status': 'Running',
        'current_iteration': 150,
        'best_solution_found': True
    })

@app.route('/api/history/list', methods=['GET'])
def list_history_files():
    history_dir = os.path.join(app.static_folder, 'js', 'history')
    try:
        files = [f for f in os.listdir(history_dir) if f.endswith('.json')]
        return jsonify(files)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<filename>', methods=['GET'])
def get_history_file(filename):
    history_dir = os.path.join(app.static_folder, 'js', 'history')
    try:
        with open(os.path.join(history_dir, filename), 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True)