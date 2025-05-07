from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
import os
from datetime import datetime
import json
import secrets  # for better secret key generation

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
    return render_template('search.html')

if __name__ == '__main__':
    app.run(debug=True)

