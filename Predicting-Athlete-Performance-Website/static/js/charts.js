// Main charts initialization
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize charts on the overview page
    if (document.getElementById('intensityVsPerformanceChart')) {
        // Initialize intensity vs performance chart
        initIntensityVsPerformanceChart();
        
        // Initialize the prediction charts
        initInjuryRiskChart();
        initPerformanceChart();
        initFatigueChart();
        
        // Setup event listeners for chart controls
        setupChartControls();
    }
    
    // Initialize Schedule Details tab
    if (document.getElementById('combinedMetricsChart')) {
        // Initialize the combined metrics chart
        initCombinedMetricsChart();
        
        // Initialize the risk heatmap
        initRiskHeatmap();
        
        // Initialize the metrics table
        initMetricsTable();
        
        // Setup Schedule tab event listeners
        setupScheduleControls();
    }
    
    // Initialize Predictions tab
    if (document.getElementById('injuryRiskPredictionChart')) {
        // Initialize prediction charts
        initInjuryRiskPredictionChart();
        initFatiguePredictionChart();
        
        // Initialize relationship analysis charts
        initIntensityVsRiskChart();
        initIntensityVsFatigueChart();
        
        // Setup predictions controls
        setupPredictionsControls();
    }
    
    // Initialize the History/Logs tab
    if (document.getElementById('historyViewSelector')) {
        initializeHistoryTab();
    }
});

// Chart color palette
const chartColors = {
    primary: '#4F7942',
    primaryLight: 'rgba(79, 121, 66, 0.2)',
    primaryMedium: 'rgba(79, 121, 66, 0.5)',
    secondary: '#FF9800',
    secondaryLight: 'rgba(255, 152, 0, 0.2)',
    danger: '#F44336',
    dangerLight: 'rgba(244, 67, 54, 0.2)',
    info: '#2196F3',
    infoLight: 'rgba(33, 150, 243, 0.2)',
    success: '#4CAF50',
    successLight: 'rgba(76, 175, 80, 0.2)',
    warning: '#FFC107',
    warningLight: 'rgba(255, 193, 7, 0.2)',
    gray: '#A9A9A9',
    grayLight: 'rgba(169, 169, 169, 0.2)',
    white: '#FFFFFF',
    black: '#000000'
};

// Global chart options
const globalChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
        duration: 1500,
        easing: 'easeOutQuart'
    },
    plugins: {
        legend: {
            labels: {
                font: {
                    family: "'Poppins', sans-serif",
                    size: 12
                }
            }
        },
        tooltip: {
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            titleColor: '#333',
            bodyColor: '#666',
            borderColor: '#ddd',
            borderWidth: 1,
            cornerRadius: 8,
            boxPadding: 6,
            usePointStyle: true,
            titleFont: {
                family: "'Poppins', sans-serif",
                size: 14,
                weight: 600
            },
            bodyFont: {
                family: "'Poppins', sans-serif",
                size: 13
            },
            padding: 10,
            boxWidth: 8
        }
    }
};

// Add Chart.js plugin for center text
Chart.register({
    id: 'centerText',
    beforeDraw: function(chart) {
        if (chart.config.options.plugins.centerText) {
            const centerText = chart.config.options.plugins.centerText;
            const ctx = chart.ctx;
            const width = chart.width;
            const height = chart.height;
            
            ctx.save();
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = centerText.color || '#666';
            ctx.font = `${centerText.fontStyle || 'normal'} ${centerText.minFontSize || 16}px 'Poppins', sans-serif`;
            
            const centerX = width / 2;
            const centerY = height / 2;
            
            ctx.fillText(centerText.text, centerX, centerY);
            ctx.restore();
        }
    }
});

// Helper function to show a notice if no search data exists
function showNoSearchNotice(containerSelector) {
    const container = document.querySelector(containerSelector) || document.body;
    // Remove any previous notice
    const oldNotice = document.getElementById('no-search-notice');
    if (oldNotice) oldNotice.remove();
    // Create notice
    const notice = document.createElement('div');
    notice.id = 'no-search-notice';
    notice.style.background = '#fff3cd';
    notice.style.color = '#856404';
    notice.style.border = '1px solid #ffeeba';
    notice.style.padding = '1.5rem';
    notice.style.margin = '2rem auto';
    notice.style.borderRadius = '8px';
    notice.style.maxWidth = '500px';
    notice.style.textAlign = 'center';
    notice.style.fontFamily = "'Poppins', sans-serif";
    notice.innerHTML = `
        <strong>No search has been performed yet.</strong><br>
        Please <a href="/search" style="color: #4F7942; text-decoration: underline; font-weight: bold;">run a search</a> to see your performance, schedule, and predictions.
    `;
    container.prepend(notice);
}

// Helper function to load current search data (with notice)
async function loadCurrentSearchDataWithNotice(containerSelector) {
    try {
        const response = await fetch('/static/js/current/current_search.json');
        if (!response.ok) throw new Error('No search data');
        const data = await response.json();
        // Remove notice if present
        const oldNotice = document.getElementById('no-search-notice');
        if (oldNotice) oldNotice.remove();
        return data;
    } catch (error) {
        showNoSearchNotice(containerSelector);
        return null;
    }
}

// Helper function to get dates based on search timestamp
function getDatesFromSearch(searchData) {
    const searchDate = new Date(searchData.timestamp);
    const days = searchData.goal_state.days;
    const dates = [];
    
    for (let i = 0; i < days; i++) {
        const date = new Date(searchDate);
        date.setDate(date.getDate() + i);
        dates.push(formatDate(date));
    }
    
    return dates;
}

// Helper function to get today's metrics
function getTodayMetrics(searchData) {
    const searchDate = new Date(searchData.timestamp);
    const today = new Date();
    const diffTime = Math.abs(today - searchDate);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    // If we're beyond the search period, return the last day's metrics
    if (diffDays >= searchData.goal_state.days) {
        return searchData.result.schedule[searchData.result.schedule.length - 1];
    }
    
    // Return the metrics for the current day
    return searchData.result.schedule[diffDays];
}

// Training Intensity vs Performance chart
async function initIntensityVsPerformanceChart() {
    const searchData = await loadCurrentSearchDataWithNotice('.overview-container');
    if (!searchData) return;

    const ctx = document.getElementById('intensityVsPerformanceChart').getContext('2d');
    const dates = getDatesFromSearch(searchData);
    
    const trainingIntensity = searchData.result.schedule.map(day => day.intensity);
    const performance = searchData.result.schedule.map(day => day.performance);

    const intensityPerformanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Training Intensity',
                    data: trainingIntensity,
                    backgroundColor: chartColors.primaryLight,
                    borderColor: chartColors.primary,
                    borderWidth: 2,
                    borderRadius: 6,
                    yAxisID: 'y',
                    order: 1
                },
                {
                    label: 'Performance',
                    data: performance,
                    type: 'line',
                    fill: false,
                    backgroundColor: chartColors.secondary,
                    borderColor: chartColors.secondary,
                    borderWidth: 3,
                    pointBackgroundColor: chartColors.white,
                    pointBorderColor: chartColors.secondary,
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.3,
                    yAxisID: 'y1',
                    order: 0
                }
            ]
        },
        options: {
            ...globalChartOptions,
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Intensity (0.3, 0.6, 0.9)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            if ([0.3, 0.6, 0.9].includes(value)) {
                                return value;
                            }
                            return '';
                        },
                        font: {
                            family: "'Poppins', sans-serif"
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y1: {
                    beginAtZero: true,
                    max: 10,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Performance (0-10)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        }
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                ...globalChartOptions.plugins,
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: {
                        usePointStyle: true,
                        boxWidth: 10,
                        boxHeight: 10,
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    }
                }
            }
        }
    });

    window.intensityPerformanceChart = intensityPerformanceChart;
}

// Enhanced Donut Chart Options
const enhancedDonutOptions = {
    ...globalChartOptions,
    cutout: '75%',
    rotation: -90,
    circumference: 360,
    animation: {
        animateScale: true,
        animateRotate: true,
        duration: 2000,
        delay: 100,
        easing: 'easeOutQuart'
    },
    plugins: {
        legend: {
            display: false
        },
        tooltip: {
            ...globalChartOptions.plugins.tooltip,
            callbacks: {
                label: function(context) {
                    // Different formatting based on the chart ID
                    if (context.chart.canvas.id === 'performanceChart') {
                        return context.label + ': ' + parseFloat(context.parsed).toFixed(1) + ' (0-10 scale)';
                    } else if (context.chart.canvas.id === 'fatigueChart') {
                        return context.label + ': ' + parseFloat(context.parsed).toFixed(1) + ' (0-4 scale)';
                    } else if (context.chart.canvas.id === 'injuryRiskChart') {
                        return context.label + ': ' + parseFloat(context.parsed).toFixed(2);
                    } else {
                        return context.label + ': ' + parseFloat(context.parsed).toFixed(1);
                    }
                }
            }
        }
    },
    elements: {
        arc: {
            borderWidth: 0,
            borderRadius: 5
        }
    }
};

// Injury Risk Chart (Donut)
async function initInjuryRiskChart() {
    const searchData = await loadCurrentSearchDataWithNotice('.overview-container');
    if (!searchData) return;

    const ctx = document.getElementById('injuryRiskChart').getContext('2d');
    const todayMetrics = getTodayMetrics(searchData);
    
    // Current injury risk
    const injuryRisk = todayMetrics.risk;
    const riskPercentage = injuryRisk * 100;
    
    // Create gradient
    const riskGradient = ctx.createLinearGradient(0, 0, 0, 200);
    riskGradient.addColorStop(0, chartColors.danger);
    riskGradient.addColorStop(1, '#ff7b72');
    
    const injuryRiskChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Risk', 'Safe'],
            datasets: [{
                data: [injuryRisk, 1 - injuryRisk],
                backgroundColor: [
                    riskGradient,
                    '#f5f5f5'
                ],
                hoverBackgroundColor: [
                    chartColors.danger,
                    '#e9e9e9'
                ],
                hoverOffset: 5,
                borderWidth: 0
            }]
        },
        options: {
            ...enhancedDonutOptions,
            plugins: {
                ...enhancedDonutOptions.plugins,
                tooltip: {
                    ...enhancedDonutOptions.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `Risk Level: ${(context.parsed).toFixed(2)}`;
                        }
                    }
                },
                centerText: {
                    text: `${(injuryRisk * 100).toFixed(1)}%`,
                    color: '#666',
                    fontStyle: 'bold',
                    minFontSize: 16
                }
            }
        }
    });
    
    // Update DOM element with current risk value
    const valueEl = document.getElementById('injuryRiskValue');
    if (valueEl) valueEl.textContent = injuryRisk.toFixed(2);
    
    // Update risk level badge
    const riskBadge = document.querySelector('.prediction-card:nth-child(1) .metric-badge');
    if (riskBadge) {
        if (injuryRisk < 0.3) {
            riskBadge.textContent = 'Low';
            riskBadge.className = 'metric-badge low';
        } else if (injuryRisk < 0.6) {
            riskBadge.textContent = 'Medium';
            riskBadge.className = 'metric-badge medium';
        } else {
            riskBadge.textContent = 'High';
            riskBadge.className = 'metric-badge high';
        }
    }
    
    window.injuryRiskChart = injuryRiskChart;
}

// Performance Chart (Donut)
async function initPerformanceChart() {
    const searchData = await loadCurrentSearchDataWithNotice('.overview-container');
    if (!searchData) return;

    const ctx = document.getElementById('performanceChart').getContext('2d');
    const todayMetrics = getTodayMetrics(searchData);
    
    // Current performance level
    const performanceLevel = todayMetrics.performance;
    const performancePercentage = (performanceLevel / 10) * 100;
    
    // Create gradient
    const performanceGradient = ctx.createLinearGradient(0, 0, 0, 200);
    performanceGradient.addColorStop(0, chartColors.primary);
    performanceGradient.addColorStop(1, '#6b9362');
    
    const performanceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Performance', 'Gap'],
            datasets: [{
                data: [performancePercentage, 100 - performancePercentage],
                backgroundColor: [
                    performanceGradient,
                    '#f5f5f5'
                ],
                hoverBackgroundColor: [
                    chartColors.primary,
                    '#e9e9e9'
                ],
                hoverOffset: 5,
                borderWidth: 0
            }]
        },
        options: {
            ...enhancedDonutOptions,
            plugins: {
                ...enhancedDonutOptions.plugins,
                tooltip: {
                    ...enhancedDonutOptions.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `Performance: ${(context.parsed / 100 * 10).toFixed(1)}/10`;
                        }
                    }
                },
                centerText: {
                    text: `${performancePercentage.toFixed(1)}%`,
                    color: '#666',
                    fontStyle: 'bold',
                    minFontSize: 16
                }
            }
        }
    });
    
    // Update DOM element with current performance value
    const valueEl = document.getElementById('performanceValue');
    if (valueEl) valueEl.textContent = performanceLevel.toFixed(1);
    
    // Update performance badge
    const perfBadge = document.querySelector('.prediction-card:nth-child(2) .metric-badge');
    if (perfBadge) {
        if (performanceLevel >= 8) {
            perfBadge.textContent = 'High';
            perfBadge.className = 'metric-badge high';
        } else if (performanceLevel >= 6) {
            perfBadge.textContent = 'Medium';
            perfBadge.className = 'metric-badge medium';
        } else {
            perfBadge.textContent = 'Low';
            perfBadge.className = 'metric-badge low';
        }
    }
    
    window.performanceChart = performanceChart;
}

// Fatigue Chart (Donut)
async function initFatigueChart() {
    const searchData = await loadCurrentSearchDataWithNotice('.overview-container');
    if (!searchData) return;

    const ctx = document.getElementById('fatigueChart').getContext('2d');
    const todayMetrics = getTodayMetrics(searchData);
    
    // Current fatigue level
    const fatigueLevel = todayMetrics.fatigue;
    const fatiguePercentage = (fatigueLevel / 4) * 100;
    
    // Create gradient
    const fatigueGradient = ctx.createLinearGradient(0, 0, 0, 200);
    fatigueGradient.addColorStop(0, chartColors.info);
    fatigueGradient.addColorStop(1, '#64b5f6');
    
    const fatigueChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Fatigue', 'Freshness'],
            datasets: [{
                data: [fatiguePercentage, 100 - fatiguePercentage],
                backgroundColor: [
                    fatigueGradient,
                    '#f5f5f5'
                ],
                hoverBackgroundColor: [
                    chartColors.info,
                    '#e9e9e9'
                ],
                hoverOffset: 5,
                borderWidth: 0
            }]
        },
        options: {
            ...enhancedDonutOptions,
            plugins: {
                ...enhancedDonutOptions.plugins,
                tooltip: {
                    ...enhancedDonutOptions.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `Fatigue Level: ${(context.parsed / 100 * 4).toFixed(1)}/4`;
                        }
                    }
                },
                centerText: {
                    text: `${fatiguePercentage.toFixed(1)}%`,
                    color: '#666',
                    fontStyle: 'bold',
                    minFontSize: 16
                }
            }
        }
    });
    
    // Update DOM element with current fatigue value
    const valueEl = document.getElementById('fatigueValue');
    if (valueEl) valueEl.textContent = fatigueLevel.toFixed(1);
    
    // Update fatigue badge
    const fatigueBadge = document.querySelector('.prediction-card:nth-child(3) .metric-badge');
    if (fatigueBadge) {
        if (fatigueLevel < 1.5) {
            fatigueBadge.textContent = 'Low';
            fatigueBadge.className = 'metric-badge low';
        } else if (fatigueLevel < 2.5) {
            fatigueBadge.textContent = 'Medium';
            fatigueBadge.className = 'metric-badge medium';
        } else {
            fatigueBadge.textContent = 'High';
            fatigueBadge.className = 'metric-badge high';
        }
    }
    
    window.fatigueChart = fatigueChart;
}

// Helper function to get last N days formatted as strings
function getLastNDays(n) {
    const dates = [];
    for (let i = n - 1; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        dates.push(formatDate(date));
    }
    return dates;
}

// Helper function to get next N days formatted as strings
function getNextNDays(n) {
    const dates = [];
    for (let i = 0; i < n; i++) {
        const date = new Date();
        date.setDate(date.getDate() + i);
        dates.push(formatDate(date));
    }
    return dates;
}

// Format date as MMM DD (e.g., May 1)
function formatDate(date) {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${months[date.getMonth()]} ${date.getDate()}`;
}

// Initialize Injury Risk Prediction Chart
async function initInjuryRiskPredictionChart() {
    const searchData = await loadCurrentSearchDataWithNotice('.predictions-container');
    if (!searchData) return;

    const ctx = document.getElementById('injuryRiskPredictionChart').getContext('2d');
    
    // Get risk data from search results
    const riskData = searchData.result.schedule.map(day => day.risk);
    const dates = getDatesFromSearch(searchData);
    
    const baseRisk = riskData[0]; // Current risk level
    const peakRisk = Math.max(...riskData);
    const avgRisk = riskData.reduce((a, b) => a + b, 0) / riskData.length;
    
    document.getElementById('currentRiskValue').textContent = baseRisk.toFixed(2);
    document.getElementById('peakRiskValue').textContent = peakRisk.toFixed(2);
    document.getElementById('avgRiskValue').textContent = avgRisk.toFixed(2);
    
    const riskLevelBadge = document.querySelector('.risk-level');
    if (peakRisk < 0.3) {
        riskLevelBadge.textContent = 'Low Risk Period';
        riskLevelBadge.className = 'risk-level low';
    } else if (peakRisk < 0.45) {
        riskLevelBadge.textContent = 'Moderate Risk Period';
        riskLevelBadge.className = 'risk-level medium';
    } else {
        riskLevelBadge.textContent = 'High Risk Period';
        riskLevelBadge.className = 'risk-level high';
    }
    
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(244, 67, 54, 0.4)');
    gradient.addColorStop(1, 'rgba(244, 67, 54, 0.0)');
    
    const riskPredictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Predicted Injury Risk',
                data: riskData,
                borderColor: chartColors.danger,
                backgroundColor: gradient,
                borderWidth: 3,
                pointBackgroundColor: chartColors.white,
                pointBorderColor: chartColors.danger,
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            ...globalChartOptions,
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Risk Level (0-1)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        },
                        callback: function(value) {
                            return value.toFixed(2);
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            },
            plugins: {
                ...globalChartOptions.plugins,
                annotation: {
                    annotations: {
                        thresholdLine1: {
                            type: 'line',
                            yMin: 0.4,
                            yMax: 0.4,
                            borderColor: 'rgba(255, 152, 0, 0.5)',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {
                                backgroundColor: 'rgba(255, 152, 0, 0.8)',
                                content: 'Moderate Risk Threshold',
                                enabled: true,
                                position: 'start'
                            }
                        },
                        thresholdLine2: {
                            type: 'line',
                            yMin: 0.65,
                            yMax: 0.65,
                            borderColor: 'rgba(244, 67, 54, 0.5)',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {
                                backgroundColor: 'rgba(244, 67, 54, 0.8)',
                                content: 'High Risk Threshold',
                                enabled: true,
                                position: 'end'
                            }
                        }
                    }
                }
            }
        }
    });
    
    window.injuryRiskPredictionChart = riskPredictionChart;
}

// Initialize Fatigue Prediction Chart
async function initFatiguePredictionChart() {
    const searchData = await loadCurrentSearchDataWithNotice('.predictions-container');
    if (!searchData) return;

    const ctx = document.getElementById('fatiguePredictionChart').getContext('2d');
    
    // Get fatigue data from search results
    const fatigueData = searchData.result.schedule.map(day => day.fatigue);
    const dates = getDatesFromSearch(searchData);
    
    const baseFatigue = fatigueData[0];
    const peakFatigue = Math.max(...fatigueData);
    const avgFatigue = fatigueData.reduce((a, b) => a + b, 0) / fatigueData.length;
    
    document.getElementById('currentFatigueValue').textContent = baseFatigue.toFixed(1);
    document.getElementById('peakFatigueValue').textContent = peakFatigue.toFixed(1);
    document.getElementById('avgFatigueValue').textContent = avgFatigue.toFixed(1);
    
    const fatigueLevelBadge = document.querySelector('.fatigue-level');
    if (peakFatigue < 1.5) {
        fatigueLevelBadge.textContent = 'Low Fatigue';
        fatigueLevelBadge.className = 'fatigue-level low';
    } else if (peakFatigue < 2.5) {
        fatigueLevelBadge.textContent = 'Moderate Fatigue';
        fatigueLevelBadge.className = 'fatigue-level medium';
    } else {
        fatigueLevelBadge.textContent = 'High Fatigue';
        fatigueLevelBadge.className = 'fatigue-level high';
    }
    
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(33, 150, 243, 0.4)');
    gradient.addColorStop(1, 'rgba(33, 150, 243, 0.0)');
    
    const fatiguePredictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Predicted Fatigue Level (0-4)',
                data: fatigueData,
                borderColor: chartColors.info,
                backgroundColor: gradient,
                borderWidth: 3,
                pointBackgroundColor: chartColors.white,
                pointBorderColor: chartColors.info,
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            ...globalChartOptions,
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 4,
                    title: {
                        display: true,
                        text: 'Fatigue Level (0-4)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        }
                    }
                }
            }
        }
    });
    
    window.fatiguePredictionChart = fatiguePredictionChart;
}

// Initialize Intensity vs Risk Chart
async function initIntensityVsRiskChart() {
    const searchData = await loadCurrentSearchDataWithNotice('.predictions-container');
    if (!searchData) return;

    const ctx = document.getElementById('intensityVsRiskChart').getContext('2d');
    
    // Get intensity and risk data from search results
    const scatterData = searchData.result.schedule.map(day => ({
        x: day.intensity,
        y: day.risk
    }));
    
    const intensityVsRiskChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Training Sessions',
                    data: scatterData,
                    backgroundColor: 'rgba(244, 67, 54, 0.7)',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }
            ]
        },
        options: {
            ...globalChartOptions,
            scales: {
                x: {
                    min: 0.2,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Training Intensity (0.3, 0.6, 0.9)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        },
                        callback: function(value) {
                            if ([0.3, 0.6, 0.9].includes(value)) {
                                return value;
                            }
                            return '';
                        }
                    }
                },
                y: {
                    min: 0,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Injury Risk (0-1)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        },
                        callback: function(value) {
                            return value.toFixed(2);
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            },
            plugins: {
                ...globalChartOptions.plugins,
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Intensity: ${context.parsed.x.toFixed(2)}, Risk: ${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });
    
    window.intensityVsRiskChart = intensityVsRiskChart;
    
    // Update insight based on actual data
    const highIntensityDays = scatterData.filter(point => point.x >= 0.6).length;
    document.getElementById('riskInsight').textContent = 
        `Risk increases significantly when intensity exceeds 0.6 for ${highIntensityDays}+ consecutive days.`;
}

// Initialize Intensity vs Fatigue Chart
async function initIntensityVsFatigueChart() {
    const searchData = await loadCurrentSearchDataWithNotice('.predictions-container');
    if (!searchData) return;

    const ctx = document.getElementById('intensityVsFatigueChart').getContext('2d');
    
    // Get intensity and fatigue data from search results
    const scatterData = searchData.result.schedule.map(day => ({
        x: day.intensity,
        y: day.fatigue
    }));
    
    const intensityVsFatigueChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Training Sessions',
                    data: scatterData,
                    backgroundColor: 'rgba(33, 150, 243, 0.7)',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }
            ]
        },
        options: {
            ...globalChartOptions,
            scales: {
                x: {
                    min: 0.2,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Training Intensity (0.3, 0.6, 0.9)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        },
                        callback: function(value) {
                            if ([0.3, 0.6, 0.9].includes(value)) {
                                return value;
                            }
                            return '';
                        }
                    }
                },
                y: {
                    min: 0,
                    max: 4.0,
                    title: {
                        display: true,
                        text: 'Fatigue Level (0-4)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif"
                        },
                        callback: function(value) {
                            return value.toFixed(1);
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            },
            plugins: {
                ...globalChartOptions.plugins,
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Intensity: ${context.parsed.x.toFixed(2)}, Fatigue: ${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });
    
    window.intensityVsFatigueChart = intensityVsFatigueChart;
    
    // Update insight based on actual data
    const highIntensityDays = scatterData.filter(point => point.x >= 0.6).length;
    document.getElementById('fatigueInsight').textContent = 
        `Fatigue increases exponentially with training intensity. Sessions above 0.6 intensity contribute significantly more to accumulated fatigue.`;
}

// Setup Predictions Controls
function setupPredictionsControls() {
    const timeSelector = document.getElementById('predictionTimeSelector');
    if (timeSelector) {
        timeSelector.addEventListener('change', function() {
            const days = parseInt(this.value);
            updatePredictionTimeHorizon(days);
        });
    }
}

// Update prediction time horizon
function updatePredictionTimeHorizon(days) {
    initInjuryRiskPredictionChart();
    initFatiguePredictionChart();
}

// Schedule Tab Charts and Controls

// Initialize Combined Metrics Chart
async function initCombinedMetricsChart() {
    const searchData = await loadCurrentSearchDataWithNotice('.schedule-container');
    if (!searchData) return;

    const ctx = document.getElementById('combinedMetricsChart').getContext('2d');
    const dates = getDatesFromSearch(searchData);
    
    // Get metrics from search data
    const trainingIntensity = searchData.result.schedule.map(day => day.intensity);
    const performance = searchData.result.schedule.map(day => day.performance);
    const injuryRisk = searchData.result.schedule.map(day => day.risk);
    const fatigue = searchData.result.schedule.map(day => day.fatigue);
    
    const combinedMetricsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Training Intensity (0.3, 0.6, 0.9)',
                    data: trainingIntensity,
                    borderColor: chartColors.primary,
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointBackgroundColor: chartColors.white,
                    pointBorderColor: chartColors.primary,
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.3,
                    yAxisID: 'yIntensity'
                },
                {
                    label: 'Performance (0-10)',
                    data: performance,
                    borderColor: chartColors.info,
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointBackgroundColor: chartColors.white,
                    pointBorderColor: chartColors.info,
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.3,
                    yAxisID: 'yPerformance'
                },
                {
                    label: 'Injury Risk (0-1)',
                    data: injuryRisk,
                    borderColor: chartColors.danger,
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointBackgroundColor: chartColors.white,
                    pointBorderColor: chartColors.danger,
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.3,
                    yAxisID: 'yRisk'
                },
                {
                    label: 'Fatigue (0-4)',
                    data: fatigue,
                    borderColor: chartColors.warning,
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointBackgroundColor: chartColors.white,
                    pointBorderColor: chartColors.warning,
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.3,
                    yAxisID: 'yFatigue'
                }
            ]
        },
        options: {
            ...globalChartOptions,
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    }
                },
                yIntensity: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Intensity',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            if ([0.3, 0.6, 0.9].includes(value)) {
                                return value;
                            }
                            return '';
                        },
                        font: {
                            family: "'Poppins', sans-serif"
                        }
                    }
                },
                yPerformance: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    beginAtZero: true,
                    max: 10,
                    title: {
                        display: true,
                        text: 'Performance',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                },
                yRisk: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Risk (0-1)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                },
                yFatigue: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    beginAtZero: true,
                    max: 4,
                    title: {
                        display: true,
                        text: 'Fatigue (0-4)',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
    
    window.combinedMetricsChart = combinedMetricsChart;
}

// Initialize Risk Heatmap
async function initRiskHeatmap() {
    const searchData = await loadCurrentSearchDataWithNotice('.schedule-container');
    if (!searchData) return;

    const heatmapContainer = document.getElementById('riskHeatmap');
    if (!heatmapContainer) return;
    
    // Clear existing content
    heatmapContainer.innerHTML = '';
    
    // Get risk data from search results
    const riskData = searchData.result.schedule.map(day => day.risk);
    
    // Create table structure for better layout
    const table = document.createElement('table');
    table.className = 'heatmap-table';
    
    // Create header row with days
    const headerRow = document.createElement('tr');
    
    // Add empty cell for corner
    const cornerCell = document.createElement('th');
    headerRow.appendChild(cornerCell);
    
    // Add day headers
    for (let i = 1; i <= riskData.length; i++) {
        const th = document.createElement('th');
        th.textContent = `Day ${i}`;
        headerRow.appendChild(th);
    }
    
    table.appendChild(headerRow);
    
    // Risk level ranges and colors
    const riskLevels = [
        { min: 0, max: 0.3, color: 'rgba(79, 121, 66, 0.2)' },    // Low risk - Light green
        { min: 0.3, max: 0.6, color: 'rgba(79, 121, 66, 0.4)' },  // Moderate risk - Medium green
        { min: 0.6, max: 0.8, color: 'rgba(79, 121, 66, 0.6)' },  // High risk - Dark green
        { min: 0.8, max: 1.0, color: 'rgba(79, 121, 66, 0.8)' }   // Critical risk - Very dark green
    ];
    
    // Create data row
    const row = document.createElement('tr');
    
    // Add row label
    const rowLabelCell = document.createElement('td');
    rowLabelCell.textContent = 'Risk Level';
    row.appendChild(rowLabelCell);
    
    // Add day cells with actual risk data
    riskData.forEach(risk => {
        const cell = document.createElement('td');
        
        // Find the appropriate risk level and color
        const riskLevel = riskLevels.find(level => risk >= level.min && risk < level.max);
        const bgColor = riskLevel ? riskLevel.color : 'rgba(79, 121, 66, 0.2)';
        
        cell.style.backgroundColor = bgColor;
        
        // Add tooltip with risk info
        cell.title = `Risk Level: ${risk.toFixed(2)}`;
        
        row.appendChild(cell);
    });
    
    table.appendChild(row);
    
    heatmapContainer.appendChild(table);
}

// Initialize Metrics Table
async function initMetricsTable() {
    const searchData = await loadCurrentSearchDataWithNotice('.schedule-container');
    if (!searchData) return;

    const tableElement = document.getElementById('metricsTable');
    if (!tableElement) return;
    
    // Create the table structure first
    tableElement.innerHTML = `
        <thead>
            <tr>
                <th>Date</th>
                <th>Intensity<br><small>(0.3, 0.6, 0.9)</small></th>
                <th>Performance<br><small>(0-10)</small></th>
                <th>Risk<br><small>(0-1)</small></th>
                <th>Fatigue<br><small>(0-4)</small></th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;
    
    const tableBody = tableElement.querySelector('tbody');
    
    // Clear existing rows
    tableBody.innerHTML = '';
    
    // Get dates and metrics from search data
    const dates = getDatesFromSearch(searchData);
    const schedule = searchData.result.schedule;
    
    // Add rows for each day
    schedule.forEach((day, index) => {
        const row = document.createElement('tr');
        
        row.innerHTML = `
            <td>${dates[index]}</td>
            <td>${day.intensity}</td>
            <td>${day.performance.toFixed(1)}</td>
            <td>${day.risk.toFixed(2)}</td>
            <td>${day.fatigue.toFixed(1)}</td>
            <td class="actions-cell">
                <button class="table-action-btn view" title="View Details">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="table-action-btn edit" title="Edit Data">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="table-action-btn delete" title="Delete Record">
                    <i class="fas fa-trash-alt"></i>
                </button>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Setup pagination
    const pageIndicator = document.getElementById('pageIndicator');
    if (pageIndicator) {
        pageIndicator.textContent = 'Page 1 of 1';
    }
    
    // Add event listeners to buttons
    const actionButtons = document.querySelectorAll('.table-action-btn');
    actionButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const action = this.classList.contains('view') ? 'View' : 
                          this.classList.contains('edit') ? 'Edit' : 'Delete';
            
            const row = this.closest('tr');
            const date = row.cells[0].textContent;
            
            alert(`${action} data for ${date}`);
        });
    });
}

// Setup Schedule Controls
function setupScheduleControls() {
    // Time range selector
    const timeSelector = document.getElementById('scheduleTimeRangeSelector');
    if (timeSelector) {
        timeSelector.addEventListener('change', function() {
            // In a real app, this would fetch new data based on the selected range
            // For demo, just reinitialize with random data
            
            // Show loading animation
            const refreshBtn = document.getElementById('refreshScheduleChart');
            if (refreshBtn) refreshBtn.classList.add('rotating');
            
            setTimeout(function() {
                initCombinedMetricsChart();
                initRiskHeatmap();
                initMetricsTable();
                if (refreshBtn) refreshBtn.classList.remove('rotating');
            }, 800);
        });
    }
    
    // Download chart button
    const downloadBtn = document.getElementById('downloadScheduleChart');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            const canvas = document.getElementById('combinedMetricsChart');
            const link = document.createElement('a');
            link.download = 'performance-metrics.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        });
    }
    
    // Refresh chart button
    const refreshBtn = document.getElementById('refreshScheduleChart');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            this.classList.add('rotating');
            
            setTimeout(() => {
                initCombinedMetricsChart();
                this.classList.remove('rotating');
            }, 800);
        });
    }
    
    // Export table data button
    const exportBtn = document.getElementById('exportTableData');
    if (exportBtn) {
        exportBtn.addEventListener('click', function() {
            alert('Data would be exported as CSV file in a real application');
        });
    }
    
    // Refresh table data button
    const refreshTableBtn = document.getElementById('refreshTableData');
    if (refreshTableBtn) {
        refreshTableBtn.addEventListener('click', function() {
            this.classList.add('rotating');
            
            setTimeout(() => {
                initMetricsTable();
                this.classList.remove('rotating');
            }, 800);
        });
    }
    
    // Pagination buttons
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');
    
    if (prevBtn && nextBtn) {
        prevBtn.addEventListener('click', function() {
            alert('Would show previous page of results in a real application');
        });
        
        nextBtn.addEventListener('click', function() {
            alert('Would show next page of results in a real application');
        });
    }
}

// Set up chart controls for the Overview page
function setupChartControls() {
    // Time range selector on overview page
    const timeSelector = document.getElementById('timeRangeSelector');
    if (timeSelector) {
        timeSelector.addEventListener('change', function() {
            // Simulate loading new data
            const refreshBtn = document.getElementById('refreshIntensityChart');
            if (refreshBtn) refreshBtn.classList.add('rotating');
            
            setTimeout(() => {
                initIntensityVsPerformanceChart();
                if (refreshBtn) refreshBtn.classList.remove('rotating');
            }, 800);
        });
    }
    
    // Download intensity chart
    const downloadBtn = document.getElementById('downloadIntensityChart');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            const canvas = document.getElementById('intensityVsPerformanceChart');
            const link = document.createElement('a');
            link.download = 'intensity-vs-performance.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        });
    }
    
    // Refresh intensity chart
    const refreshBtn = document.getElementById('refreshIntensityChart');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            this.classList.add('rotating');
            
            setTimeout(() => {
                initIntensityVsPerformanceChart();
                this.classList.remove('rotating');
            }, 800);
        });
    }
}

// History/Logs Tab Functionality
function initializeHistoryTab() {
    // View selector
    const viewSelector = document.getElementById('historyViewSelector');
    const gridView = document.querySelector('.history-grid-view');
    const listView = document.querySelector('.history-list-view');
    
    // View toggle functionality
    viewSelector.addEventListener('change', function() {
        const selectedView = this.value;
        
        if (selectedView === 'grid') {
            gridView.style.display = 'block';
            listView.style.display = 'none';
        } else if (selectedView === 'list') {
            gridView.style.display = 'none';
            listView.style.display = 'block';
        } else if (selectedView === 'timeline') {
            // Timeline view is not yet implemented
            gridView.style.display = 'none';
            listView.style.display = 'block';
            alert('Timeline view is coming soon!');
            viewSelector.value = 'list'; // Default back to list view
        }
    });
    
    // Grid list navigation
    const prevGridBtn = document.getElementById('prevGridList');
    const nextGridBtn = document.getElementById('nextGridList');
    const currentGridDisplay = document.getElementById('currentGridDisplay');
    
    let currentList = 1;
    const totalLists = 5; // This would be dynamic in a real app
    
    prevGridBtn.addEventListener('click', function() {
        if (currentList > 1) {
            currentList--;
            updateGridList();
        }
    });
    
    nextGridBtn.addEventListener('click', function() {
        if (currentList < totalLists) {
            currentList++;
            updateGridList();
        }
    });
    
    function updateGridList() {
        currentGridDisplay.textContent = `List ${currentList}`;
        
        // In a real app, you would now update the grid items for the new list
        // For this demo, we're using static grid items
        
        // Enable/disable navigation buttons
        prevGridBtn.disabled = currentList === 1;
        nextGridBtn.disabled = currentList === totalLists;
    }
    
    // Initialize grid pagination
    updateGridList();
    
    // Grid page pagination
    const prevGridPageBtn = document.getElementById('prevGridPage');
    const nextGridPageBtn = document.getElementById('nextGridPage');
    const gridPageIndicator = document.getElementById('gridPageIndicator');
    
    let currentGridPage = 1;
    const totalGridPages = 3; // This would come from backend data in a real app
    
    prevGridPageBtn.addEventListener('click', function() {
        if (currentGridPage > 1) {
            currentGridPage--;
            updateGridPagination();
        }
    });
    
    nextGridPageBtn.addEventListener('click', function() {
        if (currentGridPage < totalGridPages) {
            currentGridPage++;
            updateGridPagination();
        }
    });
    
    function updateGridPagination() {
        gridPageIndicator.textContent = `Page ${currentGridPage} of ${totalGridPages}`;
        prevGridPageBtn.disabled = currentGridPage === 1;
        nextGridPageBtn.disabled = currentGridPage === totalGridPages;
        
        // In a real app, you would now update the grid items for the new page
    }
    
    // Modal functionality
    const modal = document.getElementById('logDetailsModal');
    const closeModalBtn = document.querySelector('.close-modal-btn');
    
    // Grid item click handler
    const gridItems = document.querySelectorAll('.grid-item.has-activity');
    gridItems.forEach(item => {
        item.addEventListener('click', function() {
            const logId = this.getAttribute('data-log-id');
            showLogModal(logId);
        });
    });
    
    // List view log buttons
    const viewLogButtons = document.querySelectorAll('.view-log-btn');
    viewLogButtons.forEach(button => {
        button.addEventListener('click', function() {
            const logId = this.getAttribute('data-log-id');
            showLogModal(logId);
        });
    });
    
    // Close modal
    closeModalBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });
    
    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // Export log
    document.getElementById('exportLogBtn').addEventListener('click', function() {
        alert('Log data would be exported as CSV or JSON in a real application');
    });
    
    // Re-run algorithm
    document.getElementById('rerunAlgorithmBtn').addEventListener('click', function() {
        alert('Algorithm would be re-run with the same parameters in a real application');
        modal.style.display = 'none';
    });
    
    // Filter controls in list view
    const algorithmTypeFilter = document.getElementById('algorithmTypeFilter');
    if (algorithmTypeFilter) {
        algorithmTypeFilter.addEventListener('change', function() {
            filterHistoryList();
        });
    }
    
    const dateRangeFilter = document.getElementById('dateRangeFilter');
    if (dateRangeFilter) {
        dateRangeFilter.addEventListener('change', function() {
            filterHistoryList();
        });
    }
    
    // Pagination in list view
    const prevLogPageBtn = document.getElementById('prevLogPage');
    const nextLogPageBtn = document.getElementById('nextLogPage');
    const logPageIndicator = document.getElementById('logPageIndicator');
    
    let currentLogPage = 1;
    const totalLogPages = 3; // This would come from backend data in a real app
    
    if (prevLogPageBtn && nextLogPageBtn) {
        prevLogPageBtn.addEventListener('click', function() {
            if (currentLogPage > 1) {
                currentLogPage--;
                updateLogPagination();
            }
        });
        
        nextLogPageBtn.addEventListener('click', function() {
            if (currentLogPage < totalLogPages) {
                currentLogPage++;
                updateLogPagination();
            }
        });
    }
    
    function updateLogPagination() {
        if (logPageIndicator) {
            logPageIndicator.textContent = `Page ${currentLogPage} of ${totalLogPages}`;
        }
        
        if (prevLogPageBtn) {
            prevLogPageBtn.disabled = currentLogPage === 1;
        }
        
        if (nextLogPageBtn) {
            nextLogPageBtn.disabled = currentLogPage === totalLogPages;
        }
        
        // In a real app, you would now update the history items for the new page
    }
    
    // Function to show modal with log details
    function showLogModal(logId) {
        // In a real app, you would fetch the actual log data based on logId
        // For demo, we'll just update the modal with static data
        
        document.getElementById('modalTitle').textContent = `Log Details #${logId}`;
        
        // Simulate different algorithms based on log ID
        const algorithms = ['A* Search', 'BFS Search', 'DFS Search', 'Genetic Algorithm', 'CSP Solver'];
        const logIdNum = parseInt(logId.substr(-2));
        const algorithmIndex = logIdNum % algorithms.length;
        
        document.getElementById('logAlgorithm').textContent = algorithms[algorithmIndex];
        
        // Format date and time from the log ID
        const month = logId.substring(4, 6);
        const day = logId.substring(6, 8);
        document.getElementById('logDateTime').textContent = `May ${day}, 2025 - 14:35`;
        
        // Show the modal
        modal.style.display = 'block';
    }
    
    // Filter history list based on selected filters
    function filterHistoryList() {
        const algorithmType = algorithmTypeFilter.value;
        const dateRange = dateRangeFilter.value;
        
        // In a real app, you would filter the history items based on these values
        console.log(`Filtering by algorithm: ${algorithmType}, date range: ${dateRange}`);
        
        // For demo purposes, just show a message
        const historyItems = document.querySelectorAll('.history-item');
        historyItems.forEach(item => {
            if (algorithmType !== 'all') {
                // This is just for demonstration
                // In a real app, each item would have a data attribute with its algorithm type
                if (Math.random() > 0.5) {
                    item.style.display = 'none';
                } else {
                    item.style.display = 'flex';
                }
            } else {
                item.style.display = 'flex';
            }
        });
    }
}

// ================= EDIT HISTORY TAB =================

// Helper: Fetch list of history JSON files
async function fetchHistoryFiles() {
    try {
        const response = await fetch('/api/history/list');
        const files = await response.json();
        return files;
    } catch (e) {
        console.error('Could not fetch history files:', e);
        return [];
    }
}

// Helper: Load a single history JSON file
async function loadHistoryFile(filename) {
    try {
        const response = await fetch(`/api/history/${filename}`);
        return await response.json();
    } catch (e) {
        console.error('Could not load history file:', filename, e);
        return null;
    }
}

// Render Edit History List
async function renderEditHistoryList() {
    const container = document.getElementById('editHistoryList');
    if (!container) return;
    container.innerHTML = '<div>Loading...</div>';
    const files = await fetchHistoryFiles();
    if (!files.length) {
        container.innerHTML = '<div>No edit/search history found.</div>';
        return;
    }
    container.innerHTML = '';
    for (const file of files.reverse()) { // newest first
        const data = await loadHistoryFile(file);
        if (!data) continue;
        const div = document.createElement('div');
        div.className = 'edit-history-entry';
        div.innerHTML = `
            <strong>${data.timestamp ? new Date(data.timestamp).toLocaleString() : file}</strong>
            <span>Algorithm: ${data.algorithm || 'N/A'}</span>
            <span>Goal: ${data.goal_state ? JSON.stringify(data.goal_state) : ''}</span>
            <button class="view-log-btn" data-filename="${file}">View Details</button>
        `;
        container.appendChild(div);
    }
    // Attach click handlers
    container.querySelectorAll('.view-log-btn').forEach(btn => {
        btn.addEventListener('click', async function() {
            const filename = this.getAttribute('data-filename');
            const data = await loadHistoryFile(filename);
            showLogModalFromData(data, filename);
        });
    });
}

// Show modal with real data
function showLogModalFromData(data, filename) {
    document.getElementById('modalTitle').textContent = 'Edit/Search Log Details';
    document.getElementById('logAlgorithm').textContent = data.algorithm || 'N/A';
    document.getElementById('logDateTime').textContent = data.timestamp ? new Date(data.timestamp).toLocaleString() : '';
    document.getElementById('logParams').textContent = data.search_params ? JSON.stringify(data.search_params) : '';
    document.getElementById('logSummary').textContent = data.summary || '';
    // Fill results table
    const tbody = document.querySelector('#logResultsTable tbody');
    tbody.innerHTML = '';
    if (data.result && data.result.schedule) {
        data.result.schedule.forEach((day, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${idx + 1}</td>
                <td>${day.intensity}</td>
                <td>${day.performance.toFixed(1)}</td>
                <td>${day.risk.toFixed(2)}</td>
                <td>${day.fatigue.toFixed(1)}</td>
            `;
            tbody.appendChild(tr);
        });
    }
    // Export button
    const exportBtn = document.getElementById('exportLogBtn');
    exportBtn.onclick = function() {
        const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || 'log.json';
        a.click();
        URL.revokeObjectURL(url);
    };
    // Show modal
    document.getElementById('logDetailsModal').style.display = 'block';
}

// Close modal
const closeModalBtn = document.querySelector('.close-modal-btn');
if (closeModalBtn) {
    closeModalBtn.onclick = function() {
        document.getElementById('logDetailsModal').style.display = 'none';
    };
}

// Auto-render edit history if section exists
if (document.getElementById('editHistoryList')) {
    renderEditHistoryList();
}

// ================= ACTIVITY HISTORY & ALGORITHM LOGS (DYNAMIC) =================

// Render Activity History Grid and List Views
async function renderActivityHistory() {
    const files = await fetchHistoryFiles();
    if (!files.length) return;

    // Algorithm color mapping
    const algorithmColors = {
        'astar': 'astar',
        'bfs': 'bfs',
        'dfs': 'dfs',
        'genetic': 'genetic',
        'csp': 'csp',
        'greedy': 'astar', // Using astar color for greedy as it's not in the legend
        'ucs': 'bfs'      // Using bfs color for ucs as it's not in the legend
    };

    // Grid view
    const grid = document.querySelector('.log-grid');
    if (grid) {
        grid.innerHTML = '';
        files.reverse().forEach((file, idx) => {
            loadHistoryFile(file).then(data => {
                if (!data) return;
                const div = document.createElement('div');
                div.className = `grid-item has-activity ${algorithmColors[data.algorithm] || 'astar'}`;
                div.setAttribute('data-log-id', file);
                div.innerHTML = `
                    <div class="log-id">${idx + 1}</div>
                    <div class="activity-dot" title="${data.algorithm || 'Algorithm'}"></div>
                `;
                div.onclick = () => showLogModalFromData(data, file);
                grid.appendChild(div);
            });
        });
    }

    // List view
    const list = document.querySelector('.history-list');
    if (list) {
        list.innerHTML = '';
        files.reverse().forEach((file, idx) => {
            loadHistoryFile(file).then(data => {
                if (!data) return;
                const item = document.createElement('div');
                item.className = `history-item ${algorithmColors[data.algorithm] || 'astar'}`;
                item.setAttribute('data-log-id', file);
                const date = data.timestamp ? new Date(data.timestamp) : null;
                item.innerHTML = `
                    <div class="history-item-date">
                        <div class="date-day">${date ? date.getDate().toString().padStart(2, '0') : ''}</div>
                        <div class="date-month">${date ? date.toLocaleString('default', {month: 'short'}) : ''}</div>
                    </div>
                    <div class="history-item-content">
                        <div class="item-title">${data.algorithm || 'Algorithm'}</div>
                        <div class="item-details">${data.summary || ''}</div>
                        <div class="item-metrics">
                            <span class="metric"><i class="fas fa-clock"></i> ${date ? date.toLocaleTimeString() : ''}</span>
                        </div>
                    </div>
                    <div class="history-item-actions">
                        <button class="view-log-btn" data-log-id="${file}"><i class="fas fa-eye"></i></button>
                    </div>
                `;
                item.querySelector('.view-log-btn').onclick = () => showLogModalFromData(data, file);
                list.appendChild(item);
            });
        });
    }
}

// Auto-render activity history if grid or list exists
if (document.querySelector('.log-grid') || document.querySelector('.history-list')) {
    renderActivityHistory();
}