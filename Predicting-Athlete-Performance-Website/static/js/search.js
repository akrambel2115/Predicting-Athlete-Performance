document.addEventListener('DOMContentLoaded', function() {
    // Elements for multi-step navigation
    const progressBar = document.getElementById('search-progress-bar');
    const steps = document.querySelectorAll('.step');
    const stepContents = document.querySelectorAll('.search-step');
    
    // Elements for range sliders
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    
    // Elements for advanced options
    const advancedToggleBtn = document.getElementById('advanced-toggle-btn');
    const advancedOptions = document.getElementById('advanced-options');
    const algorithmRadios = document.querySelectorAll('input[name="algorithm"]');
    const algorithmParams = document.querySelectorAll('.algorithm-params');
    
    // Elements for search execution
    const runSearchBtn = document.getElementById('run-search-btn');
    const startOverBtn = document.getElementById('start-over-btn');
    
    // Initialize slider values
    rangeInputs.forEach(input => {
        const displayElement = document.getElementById(`${input.id}-display`);
        if (displayElement) {
            displayElement.textContent = input.value;
            
            // Update display on input change
            input.addEventListener('input', () => {
                displayElement.textContent = input.value;
            });
        }
    });
    
    // Next step buttons
    document.querySelectorAll('.btn-next').forEach(button => {
        button.addEventListener('click', () => {
            const currentStep = parseInt(button.closest('.search-step').getAttribute('data-step'));
            const nextStep = parseInt(button.getAttribute('data-next'));
            
            // Update step classes
            steps[currentStep - 1].classList.remove('active');
            steps[currentStep - 1].classList.add('completed');
            steps[nextStep - 1].classList.add('active');
            
            // Update content visibility
            stepContents[currentStep - 1].style.display = 'none';
            stepContents[nextStep - 1].style.display = 'block';
            
            // Update progress bar
            progressBar.style.width = `${nextStep * 25}%`;
        });
    });
    
    // Previous step buttons
    document.querySelectorAll('.btn-previous').forEach(button => {
        button.addEventListener('click', () => {
            const currentStep = parseInt(button.closest('.search-step').getAttribute('data-step'));
            const prevStep = parseInt(button.getAttribute('data-previous'));
            
            // Update step classes
            steps[currentStep - 1].classList.remove('active');
            steps[prevStep - 1].classList.remove('completed');
            steps[prevStep - 1].classList.add('active');
            
            // Update content visibility
            stepContents[currentStep - 1].style.display = 'none';
            stepContents[prevStep - 1].style.display = 'block';
            
            // Update progress bar
            progressBar.style.width = `${prevStep * 25}%`;
        });
    });
    
    // Advanced options toggle
    if (advancedToggleBtn) {
        advancedToggleBtn.addEventListener('click', () => {
            if (advancedOptions.style.display === 'none') {
                advancedOptions.style.display = 'block';
                advancedToggleBtn.querySelector('.fa-chevron-down').classList.replace('fa-chevron-down', 'fa-chevron-up');
            } else {
                advancedOptions.style.display = 'none';
                advancedToggleBtn.querySelector('.fa-chevron-up').classList.replace('fa-chevron-up', 'fa-chevron-down');
            }
        });
    }
    
    // Algorithm selection toggle for advanced parameters
    algorithmRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            const selectedAlgorithm = radio.value;
            
            // Hide all algorithm params
            algorithmParams.forEach(param => {
                param.style.display = 'none';
            });
            
            // Show selected algorithm params
            const selectedParams = document.getElementById(`${selectedAlgorithm}-params`);
            if (selectedParams) {
                selectedParams.style.display = 'block';
            }
        });
    });
    
    // Run search button
    if (runSearchBtn) {
        runSearchBtn.addEventListener('click', () => {
            console.log("Run search button clicked");
            
            // Collect all form inputs
            const searchParams = {
                initialState: {
                    performance: parseFloat(document.getElementById('initial-performance').value),
                    fatigue: parseFloat(document.getElementById('initial-fatigue').value),
                    risk: parseFloat(document.getElementById('initial-risk').value)
                },
                goalState: {
                    targetPerformance: parseFloat(document.getElementById('target-performance').value),
                    maxFatigue: parseFloat(document.getElementById('max-fatigue').value),
                    maxRisk: parseFloat(document.getElementById('max-risk').value),
                    days: parseInt(document.getElementById('training-days').value)
                },
                algorithm: document.querySelector('input[name="algorithm"]:checked').value,
                advancedParams: {}
            };
            
            // Add advanced params based on algorithm
            switch (searchParams.algorithm) {
                case 'astar':
                    searchParams.advancedParams.maxIterations = parseInt(document.getElementById('astar-max-iterations').value);
                    searchParams.advancedParams.timeLimit = parseInt(document.getElementById('astar-time-limit').value);
                    break;
                case 'genetic':
                    searchParams.advancedParams.populationSize = parseInt(document.getElementById('genetic-population-size').value);
                    searchParams.advancedParams.generations = parseInt(document.getElementById('genetic-generations').value);
                    searchParams.advancedParams.mutationRate = parseFloat(document.getElementById('genetic-mutation-rate').value);
                    break;
                case 'bfs':
                    searchParams.advancedParams.maxDepth = parseInt(document.getElementById('bfs-max-depth').value);
                    break;
                case 'dfs':
                    searchParams.advancedParams.maxDepth = parseInt(document.getElementById('dfs-max-depth').value);
                    break;
                case 'ucs':
                    searchParams.advancedParams.maxIterations = parseInt(document.getElementById('ucs-max-iterations').value);
                    searchParams.advancedParams.timeLimit = parseInt(document.getElementById('ucs-time-limit').value);
                    break;
                case 'greedy':
                    searchParams.advancedParams.maxIterations = parseInt(document.getElementById('greedy-max-iterations').value);
                    searchParams.advancedParams.timeLimit = parseInt(document.getElementById('greedy-time-limit').value);
                    break;
            }
            
            // Navigate to results step
            steps[2].classList.remove('active');
            steps[2].classList.add('completed');
            steps[3].classList.add('active');
            
            stepContents[2].style.display = 'none';
            stepContents[3].style.display = 'block';
            
            progressBar.style.width = '100%';
            
            // Show search progress
            document.getElementById('search-progress').style.display = 'block';
            document.getElementById('search-results-success').style.display = 'none';
            document.getElementById('search-results-failure').style.display = 'none';
            document.getElementById('search-status').textContent = 'Search in progress...';
            
            // Reset progress stats
            document.getElementById('nodes-explored').textContent = '0';
            document.getElementById('queue-size').textContent = '0';
            document.getElementById('elapsed-time').textContent = '0s';
            
            // Start search execution
            startSearch(searchParams);
        });
    }
    
    // Start over button
    if (startOverBtn) {
        startOverBtn.addEventListener('click', () => {
            // Reset to step 1
            steps.forEach((step, index) => {
                step.classList.remove('active', 'completed');
                if (index === 0) {
                    step.classList.add('active');
                }
            });
            
            stepContents.forEach((content, index) => {
                content.style.display = index === 0 ? 'block' : 'none';
            });
            
            progressBar.style.width = '25%';
            
            // Reset all search results and progress
            document.getElementById('search-progress').style.display = 'block';
            document.getElementById('search-results-success').style.display = 'none';
            document.getElementById('search-results-failure').style.display = 'none';
            document.getElementById('search-status').textContent = 'Search in progress...';
            
            // Reset progress stats
            document.getElementById('nodes-explored').textContent = '0';
            document.getElementById('queue-size').textContent = '0';
            document.getElementById('elapsed-time').textContent = '0s';
        });
    }
    
    // Function to start search
    function startSearch(params) {
        console.log("Starting search with parameters:", JSON.stringify(params, null, 2));
        
        // Make API call to start the search
        fetch('/api/run_search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Search completed:", data);
            
            // Update the UI based on the response
            if (data.success) {
                showSearchResults(data);
            } else {
                showSearchFailure(data);
            }
        })
        .catch(error => {
            console.error("Search error:", error);
            showSearchFailure({
                success: false,
                error: error.message,
                stats: {
                    nodesExplored: 0,
                    maxQueueSize: 0,
                    executionTime: 0
                }
            });
        });
        
        // Start polling for progress updates if we want real-time updates
        // Only needed for long-running searches
        let progressInterval = setInterval(() => {
            fetch('/api/search_progress')
                .then(response => response.json())
                .catch(error => {
                    console.error("Error fetching progress:", error);
                });
        }, 1000);  // Update every second
        
        // Clear interval after 30 seconds (or could clear it when search completes)
        setTimeout(() => {
            clearInterval(progressInterval);
        }, 30000);
    }
    
    // Show successful search results
    function showSearchResults(data) {
        // Hide progress indicator
        document.getElementById('search-progress').style.display = 'none';
        
        // Update status
        document.getElementById('search-status').textContent = 'Optimal training plan found!';
        
        // Show success results panel
        const resultsPanel = document.getElementById('search-results-success');
        resultsPanel.style.display = 'block';
        
        // Update summary stats
        document.getElementById('final-performance').textContent = data.finalState.performance.toFixed(2);
        document.getElementById('final-fatigue').textContent = data.finalState.fatigue.toFixed(2);
        document.getElementById('final-risk').textContent = data.finalState.risk.toFixed(2);
        console.log(data);
        console.log(data.schedule)
        // Update algorithm stats
        document.getElementById('stat-nodes-explored').textContent = data.stats.nodesExplored.toLocaleString();
        document.getElementById('stat-max-queue').textContent = data.stats.maxQueueSize.toLocaleString();
        document.getElementById('stat-execution-time').textContent = data.stats.executionTime.toFixed(2) + 's';
        

        // Create schedule table
        createScheduleTable(data.schedule);
        
        // Create performance chart
        createPerformanceChart(data.schedule);
    }
    
    // Show search failure
    function showSearchFailure(data) {
        // Hide progress indicator
        document.getElementById('search-progress').style.display = 'none';
        
        // Update status
        document.getElementById('search-status').textContent = 'Search completed without finding a solution';
        
        // Show failure results panel
        const failurePanel = document.getElementById('search-results-failure');
        failurePanel.style.display = 'block';
        
        // Update failure stats
        document.getElementById('fail-nodes-explored').textContent = data.stats.nodesExplored.toLocaleString();
        document.getElementById('fail-max-queue').textContent = data.stats.maxQueueSize.toLocaleString();
        document.getElementById('fail-execution-time').textContent = data.stats.executionTime.toFixed(2) + 's';
    }
    
    // Create schedule table
    function createScheduleTable(schedule) {
        const tableBody = document.getElementById('schedule-body');
        tableBody.innerHTML = '';
        
        // Validate schedule data
        if (!Array.isArray(schedule)) {
            console.error('Invalid schedule data:', schedule);
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="6" class="error">No training plan data available</td>`;
            tableBody.appendChild(row);
            return;
        }

        // Handle empty schedule case
        if (schedule.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="6">No training activities recorded</td>`;
            tableBody.appendChild(row);
            return;
        }

        // Process valid schedule data
        schedule.forEach(day => {
            const row = document.createElement('tr');
            const desc = {
                '0,0.0': "Complete Rest",
                '60,0.3': "Light Recovery",
                '60,0.6': "Moderate Endurance",
                '60,0.9': "High-Intensity Short Session",
                '90,0.3': "Extended Recovery",
                '90,0.6': "Steady Training",
                '90,0.9': "Intense Conditioning",
                '120,0.3': "Long Recovery",
                '120,0.6': "Extended Training",
                '120,0.9': "Endurance Max Effort"
            };
            // Safely access properties with defaults
            const dayNumber = day.day ?? 'N/A';
            const action = day.action || [null, null];
            const intensity = day.intensity;
            const duration = day.duration;
            const performance = typeof day.performance === 'number' ? day.performance.toFixed(2) : '-';
            const fatigue = typeof day.fatigue === 'number' ? day.fatigue.toFixed(2) : '-';
            const risk = typeof day.risk === 'number' ? day.risk.toFixed(2) : '-';
            const key = `${duration},${intensity}`;

            const description = desc[key] ?? "Custom Session";

            row.innerHTML = `
                <td>${dayNumber}</td>
                <td>${intensity}</td>
                <td>${duration}</td>
                <td>${description}</td>
                <td>${performance}</td>
                <td>${fatigue}</td>
                <td>${risk}</td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    // Create performance chart
    function createPerformanceChart(schedule) {
        const ctx = document.getElementById('performance-chart').getContext('2d');
        console.log(schedule);
        // Destroy existing chart instance
        if (window.performanceChart) {
            window.performanceChart.destroy();
        }

        // Process data with validation
        const processed = schedule.map(day => ({
            day: Number(day.day) || 0,
            performance: Number(day.performance) || 0,
            fatigue: Number(day.fatigue) || 0,
            risk: Number(day.risk) || 0
        })).sort((a, b) => a.day - b.day);

        // Extract data with fallback values
        const days = processed.map(d => d.day);
        const performances = processed.map(d => d.performance);
        const fatigues = processed.map(d => d.fatigue);
        const risks = processed.map(d => d.risk);

        // Calculate dynamic axis limits
        const maxPerformance = Math.max(10, Math.ceil(Math.max(...performances) + 1));
        const maxFatigue = Math.ceil(Math.max(...fatigues) + 1);
        const maxRisk = Math.ceil(Math.max(...risks) + 0.1);

        window.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: days,
                datasets: [
                    {
                        label: 'Performance',
                        data: performances,
                        borderColor: '#4F7942',
                        backgroundColor: 'rgba(79, 121, 66, 0.1)',
                        tension: 0.3,
                        fill: true,
                        yAxisID: 'performance'
                    },
                    {
                        label: 'Fatigue',
                        data: fatigues,
                        borderColor: '#FF6B6B',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.3,
                        fill: true,
                        yAxisID: 'fatigue'
                    },
                    {
                        label: 'Injury Risk',
                        data: risks,
                        borderColor: '#FFB347',
                        backgroundColor: 'rgba(255, 179, 71, 0.1)',
                        tension: 0.3,
                        fill: true,
                        yAxisID: 'risk'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Day'
                        },
                        type: 'linear',
                        position: 'bottom'
                    },
                    performance: {
                        type: 'linear',
                        position: 'left',
                        title: { display: true, text: 'Performance' },
                        min: 0,
                        max: maxPerformance,
                        grid: { borderColor: '#4F7942' },
                        ticks: { color: '#4F7942' }
                    },
                    fatigue: {
                        type: 'linear',
                        position: 'right',
                        title: { display: true, text: 'Fatigue' },
                        min: 0,
                        max: maxFatigue,
                        grid: { display: false },
                        ticks: { color: '#FF6B6B' }
                    },
                    risk: {
                        type: 'linear',
                        position: 'right',
                        title: { display: true, text: 'Risk' },
                        min: 0,
                        max: maxRisk,
                        grid: { display: false },
                        ticks: { 
                            color: '#FFB347',
                            callback: value => value.toFixed(1)
                        }
                    }
                },
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y || 0;
                                return `${label}: ${value.toFixed(2)}`;
                            }
                        }
                    }
                }
            }
        });
    }
});