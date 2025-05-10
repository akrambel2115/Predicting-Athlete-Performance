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
        
        // Initialize optimal training zone chart
        initOptimalTrainingZoneChart();
        
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

// Training Intensity vs Performance chart
function initIntensityVsPerformanceChart() {
    const ctx = document.getElementById('intensityVsPerformanceChart').getContext('2d');

    const dates = getLastNDays(7);
    const trainingIntensity = [0.3, 0.6, 0.9, 0.3, 0.3, 0.6, 0.9];
    const performance = [7.6, 7.0, 6.5, 7.2, 7.8, 7.4, 8.0];

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
function initInjuryRiskChart() {
    const ctx = document.getElementById('injuryRiskChart').getContext('2d');
    
    // Current injury risk (0.24 on a relative scale)
    const injuryRisk = 0.24;
    const riskPercentage = injuryRisk * 100; // Convert to percentage for chart display
    
    // Create gradient
    const riskGradient = ctx.createLinearGradient(0, 0, 0, 200);
    riskGradient.addColorStop(0, chartColors.danger);
    riskGradient.addColorStop(1, '#ff7b72');
    
    const injuryRiskChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Risk', 'Safe'],
            datasets: [{
                data: [riskPercentage, 100 - riskPercentage],
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
        options: enhancedDonutOptions
    });
    
    // Draw center text with smooth animation
    let currentValue = 0;
    const targetValue = injuryRisk;
    
    const renderValue = () => {
        if (currentValue < targetValue) {
            const step = Math.max(0.01, (targetValue - currentValue) / 10);
            currentValue = Math.min(currentValue + step, targetValue);
            
            // Update DOM element
            const valueEl = document.getElementById('injuryRiskValue');
            if (valueEl) valueEl.textContent = currentValue.toFixed(2);
            
            requestAnimationFrame(renderValue);
        }
    };
    
    setTimeout(renderValue, 500); // Start a bit after chart animation begins
    
    // Draw center text in chart
    Chart.register({
        id: 'centerTextPlugin',
        afterDraw: function(chart) {
            if (chart.canvas.id === 'injuryRiskChart') {
                const width = chart.width;
                const height = chart.height;
                const ctx = chart.ctx;
                
                ctx.restore();
                const fontSize = (height / 150).toFixed(2);
                const fontFamily = "'Poppins', sans-serif";
                
                // Draw risk value
                ctx.font = `bold ${fontSize}em ${fontFamily}`;
                ctx.textBaseline = 'middle';
                ctx.textAlign = 'center';
                
                const text = currentValue.toFixed(2);
                const textY = height / 2 - 10;
                
                ctx.fillStyle = chartColors.danger;
                ctx.fillText(text, width / 2, textY);
                
                // Draw label
                ctx.font = `${fontSize * 0.45}em ${fontFamily}`;
                const subText = 'Risk Level';
                const subTextY = height / 2 + 15;
                
                ctx.fillStyle = '#888';
                ctx.fillText(subText, width / 2, subTextY);
                ctx.save();
            }
        }
    });
    
    // Store reference to chart for updates
    window.injuryRiskChart = injuryRiskChart;
}

// Performance Chart (Donut)
function initPerformanceChart() {
    const ctx = document.getElementById('performanceChart').getContext('2d');
      // Current performance level (8.3 on scale of 0-10)
    const performanceLevel = 8.3;
    const performancePercentage = (performanceLevel / 10) * 100; // Convert to percentage for chart display
    
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
        options: enhancedDonutOptions
    });
    
    // Draw center text with smooth animation
    let currentValue = 0;
    const targetValue = performanceLevel;
    
    const renderValue = () => {
        if (currentValue < targetValue) {
            const step = Math.max(0.1, (targetValue - currentValue) / 10);
            currentValue = Math.min(currentValue + step, targetValue);
            
            // Update DOM element
            const valueEl = document.getElementById('performanceValue');
            if (valueEl) valueEl.textContent = currentValue.toFixed(1);
            
            requestAnimationFrame(renderValue);
        }
    };
    
    setTimeout(renderValue, 800); // Start a bit after chart animation begins
    
    // Draw center text in chart
    Chart.register({
        id: 'centerTextPlugin2',
        afterDraw: function(chart) {
            if (chart.canvas.id === 'performanceChart') {
                const width = chart.width;
                const height = chart.height;
                const ctx = chart.ctx;
                
                ctx.restore();
                const fontSize = (height / 150).toFixed(2);
                const fontFamily = "'Poppins', sans-serif";
                
                // Draw performance value
                ctx.font = `bold ${fontSize}em ${fontFamily}`;
                ctx.textBaseline = 'middle';
                ctx.textAlign = 'center';
                
                const text = currentValue.toFixed(1);
                const textY = height / 2 - 10;
                
                ctx.fillStyle = chartColors.primary;
                ctx.fillText(text, width / 2, textY);
                
                // Draw label
                ctx.font = `${fontSize * 0.45}em ${fontFamily}`;
                const subText = 'Performance';
                const subTextY = height / 2 + 15;
                
                ctx.fillStyle = '#888';
                ctx.fillText(subText, width / 2, subTextY);
                ctx.save();
            }
        }
    });
    
    // Store reference to chart for updates
    window.performanceChart = performanceChart;
}

// Fatigue Chart (Donut)
function initFatigueChart() {
    const ctx = document.getElementById('fatigueChart').getContext('2d');
    
    // Current fatigue level (1.6 on scale of 0-4)
    const fatigueLevel = 1.6;
    const fatiguePercentage = (fatigueLevel / 4) * 100; // Convert to percentage for chart display - scale of 0-4
    
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
        options: enhancedDonutOptions
    });
    
    // Draw center text with smooth animation
    let currentValue = 0;
    const targetValue = fatigueLevel;
    
    const renderValue = () => {
        if (currentValue < targetValue) {
            const step = Math.max(0.1, (targetValue - currentValue) / 10);
            currentValue = Math.min(currentValue + step, targetValue);
            
            // Update DOM element
            const valueEl = document.getElementById('fatigueValue');
            if (valueEl) valueEl.textContent = currentValue.toFixed(1);
            
            requestAnimationFrame(renderValue);
        }
    };
    
    setTimeout(renderValue, 1100); // Start a bit after chart animation begins
    
    // Draw center text in chart
    Chart.register({
        id: 'centerTextPlugin3',
        afterDraw: function(chart) {
            if (chart.canvas.id === 'fatigueChart') {
                const width = chart.width;
                const height = chart.height;
                const ctx = chart.ctx;
                
                ctx.restore();
                const fontSize = (height / 150).toFixed(2);
                const fontFamily = "'Poppins', sans-serif";
                
                // Draw fatigue value
                ctx.font = `bold ${fontSize}em ${fontFamily}`;
                ctx.textBaseline = 'middle';
                ctx.textAlign = 'center';
                
                const text = currentValue.toFixed(1);
                const textY = height / 2 - 10;
                
                ctx.fillStyle = chartColors.info;
                ctx.fillText(text, width / 2, textY);
                
                // Draw label
                ctx.font = `${fontSize * 0.45}em ${fontFamily}`;
                const subText = 'Fatigue (0-4)';
                const subTextY = height / 2 + 15;
                
                ctx.fillStyle = '#888';
                ctx.fillText(subText, width / 2, subTextY);
                ctx.save();
            }
        }
    });
    
    // Store reference to chart for updates
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
function initInjuryRiskPredictionChart() {
    const ctx = document.getElementById('injuryRiskPredictionChart').getContext('2d');
    
    // Number of days to predict
    const days = parseInt(document.getElementById('predictionTimeSelector')?.value || 7);
    const futureDates = getNextNDays(days);
    
    // Generate simulated injury risk prediction data
    const baseRisk = 0.32; // Current risk level (0-1)
    let riskPredictions = [];
    let highRiskPeriodStart = Math.floor(Math.random() * (days - 3)) + 2;
    
    for (let i = 0; i < days; i++) {
        let dailyRisk = baseRisk + (Math.sin(i * 0.5) * 0.05) + (Math.random() * 0.06 - 0.03);
        
        if (i >= highRiskPeriodStart && i <= highRiskPeriodStart + 2) {
            dailyRisk += 0.15;
        }
        
        dailyRisk = Math.min(1.0, Math.max(0, dailyRisk));
        riskPredictions.push(dailyRisk);
    }
    
    const peakRisk = Math.max(...riskPredictions);
    const avgRisk = riskPredictions.reduce((a, b) => a + b, 0) / riskPredictions.length;
    
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
            labels: futureDates,
            datasets: [{
                label: 'Predicted Injury Risk',
                data: riskPredictions,
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
function initFatiguePredictionChart() {
    const ctx = document.getElementById('fatiguePredictionChart').getContext('2d');
    
    const days = parseInt(document.getElementById('predictionTimeSelector')?.value || 7);
    const futureDates = getNextNDays(days);
    
    // Base fatigue level on 0-4 scale
    const baseFatigue = 1.8;
    let fatiguePredictions = [];
    
    // Training pattern affects fatigue changes (values represent relative impact)
    const trainingPattern = [0.3, 0.6, 0.9, 0.3, 0.6, 0.3, 0];
    
    for (let i = 0; i < days; i++) {
        const dayInPattern = i % trainingPattern.length;
        const trainingIntensity = trainingPattern[dayInPattern];
        
        let dailyChange = 0;
        
        if (trainingIntensity === 0) {
            // Rest day reduces fatigue
            dailyChange = -(Math.random() * 0.3 + 0.2);
        } else {
            // Training increases fatigue based on intensity
            dailyChange = trainingIntensity * (Math.random() * 0.3 + 0.1);
        }
        
        let previousFatigue = i === 0 ? baseFatigue : fatiguePredictions[i-1];
        let fatigueMultiplier = previousFatigue > 3 ? 1.2 : previousFatigue > 2 ? 1 : 0.8;
        
        let dailyFatigue = previousFatigue + (dailyChange * fatigueMultiplier);
        
        // Ensure fatigue stays within 0-4 scale
        dailyFatigue = Math.min(4, Math.max(0, dailyFatigue));
        fatiguePredictions.push(dailyFatigue);
    }
    
    const peakFatigue = Math.max(...fatiguePredictions);
    const avgFatigue = fatiguePredictions.reduce((a, b) => a + b, 0) / fatiguePredictions.length;
    
    // Update display values using 0-4 scale
    document.getElementById('currentFatigueValue').textContent = baseFatigue.toFixed(1);
    document.getElementById('peakFatigueValue').textContent = peakFatigue.toFixed(1);
    document.getElementById('avgFatigueValue').textContent = avgFatigue.toFixed(1);
    
    // Update fatigue level badge
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
            labels: futureDates,
            datasets: [{
                label: 'Predicted Fatigue Level (0-4)',
                data: fatiguePredictions,
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
function initIntensityVsRiskChart() {
    const ctx = document.getElementById('intensityVsRiskChart').getContext('2d');
    
    const scatterData = [];
    
    // Generate data points for each intensity level
    const intensities = [0.3, 0.6, 0.9];
    for (const intensity of intensities) {
        // Generate multiple points for each intensity level
        for (let i = 0; i < 10; i++) {
            const risk = (Math.random() * 0.15 + 0.1).toFixed(2);
            scatterData.push({ x: intensity, y: risk });
        }
    }
    
    const trendlinePoints = [];
    for (let x = 0.3; x <= 0.9; x += 0.1) {
        const y = 0.15 * Math.pow(x, 2) + 0.1;
        trendlinePoints.push({ x, y: Math.max(0.05, Math.min(1.0, y)) });
    }
    
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
                },
                {
                    label: 'Risk Correlation',
                    data: trendlinePoints,
                    type: 'line',
                    fill: false,
                    borderColor: 'rgba(76, 175, 80, 0.7)',
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.4
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
                annotation: {
                    annotations: {
                        highRiskZone: {
                            type: 'box',
                            xMin: 0.75,
                            xMax: 1.0,
                            yMin: 0.4,
                            yMax: 1.0,
                            backgroundColor: 'rgba(244, 67, 54, 0.1)',
                            borderColor: 'rgba(244, 67, 54, 0.3)',
                            borderWidth: 1,
                            label: {
                                display: true,
                                content: 'High Risk Zone',
                                position: 'center',
                                font: {
                                    family: "'Poppins', sans-serif",
                                    size: 12
                                },
                                color: 'rgba(244, 67, 54, 0.7)'
                            }
                        }
                    }
                }
            }
        }
    });
    
    window.intensityVsRiskChart = intensityVsRiskChart;
    
    document.getElementById('riskInsight').textContent = 'Risk increases significantly when intensity exceeds 0.6 for 3+ consecutive days.';
}

// Initialize Intensity vs Fatigue Chart
function initIntensityVsFatigueChart() {
    const ctx = document.getElementById('intensityVsFatigueChart').getContext('2d');
    
    const scatterData = [];
    
    // Generate data points for each intensity level
    const intensities = [0.3, 0.6, 0.9];
    for (const intensity of intensities) {
        // Generate multiple points for each intensity level
        for (let i = 0; i < 10; i++) {
            let fatigue;
            if (intensity === 0.3) {
                fatigue = (Math.random() * 0.6 + 0.6).toFixed(1); // 0.6-1.2 on 0-4 scale
            } else if (intensity === 0.6) {
                fatigue = (Math.random() * 1.0 + 1.2).toFixed(1); // 1.2-2.2 on 0-4 scale
            } else {
                fatigue = (Math.random() * 1.4 + 2.0).toFixed(1); // 2.0-3.4 on 0-4 scale
            }
            scatterData.push({ x: intensity, y: fatigue });
        }
    }
    
    // Create trendline points showing exponential relationship
    const trendlinePoints = [];
    for (let x = 0.3; x <= 0.9; x += 0.1) {
        // Quadratic relationship: fatigue = base + a*x + b*x^2
        // This creates a more pronounced curve that better matches the scatter data
        const y = 0.4 + 1.2 * x + 1.8 * Math.pow(x, 2);
        trendlinePoints.push({ x, y: Math.max(0.4, Math.min(4.0, y)) });
    }
    
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
                },
                {
                    label: 'Fatigue Correlation',
                    data: trendlinePoints,
                    type: 'line',
                    fill: false,
                    borderColor: 'rgba(76, 175, 80, 0.7)',
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.4
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
                annotation: {
                    annotations: {
                        highFatigueZone: {
                            type: 'box',
                            xMin: 0.75,
                            xMax: 1.0,
                            yMin: 2.6,
                            yMax: 4.0,
                            backgroundColor: 'rgba(33, 150, 243, 0.1)',
                            borderColor: 'rgba(33, 150, 243, 0.3)',
                            borderWidth: 1,
                            label: {
                                display: true,
                                content: 'High Fatigue Zone',
                                position: 'center',
                                font: {
                                    family: "'Poppins', sans-serif",
                                    size: 12
                                },
                                color: 'rgba(33, 150, 243, 0.7)'
                            }
                        }
                    }
                }
            }
        }
    });
    
    window.intensityVsFatigueChart = intensityVsFatigueChart;
    
    document.getElementById('fatigueInsight').textContent = 'Fatigue increases exponentially with training intensity. Sessions above 0.6 intensity contribute significantly more to accumulated fatigue.';
}

// Initialize Optimal Training Zone Chart
function initOptimalTrainingZoneChart() {
    const ctx = document.getElementById('optimalTrainingZoneChart').getContext('2d');
    
    const intensityRange = [];
    for (let i = 0; i <= 100; i += 5) {
        intensityRange.push(i);
    }
    
    const performanceBenefit = intensityRange.map(x => {
        return 100 * Math.exp(-0.001 * Math.pow(x - 75, 2));
    });
    
    const injuryRisk = intensityRange.map(x => {
        if (x < 40) return 2;
        if (x < 60) return 5 + (x - 40) * 0.3;
        if (x < 75) return 11 + (x - 60) * 1;
        return 26 + (x - 75) * 3;
    });
    
    const fatigueAccumulation = intensityRange.map(x => {
        return Math.min(100, 10 + x * 0.7 + Math.pow(x/100, 2) * 20);
    });
    
    const benefitScore = intensityRange.map((x, i) => {
        return performanceBenefit[i] * 0.8 - injuryRisk[i] * 0.5 - fatigueAccumulation[i] * 0.3;
    });
    
    const maxBenefitScore = Math.max(...benefitScore);
    const thresholdScore = maxBenefitScore * 0.8;
    
    let optimalMin = 0, optimalMax = 0;
    for (let i = 0; i < benefitScore.length; i++) {
        if (benefitScore[i] >= thresholdScore) {
            optimalMin = intensityRange[i];
            break;
        }
    }
    
    for (let i = benefitScore.length - 1; i >= 0; i--) {
        if (benefitScore[i] >= thresholdScore) {
            optimalMax = intensityRange[i];
            break;
        }
    }
    
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(79, 121, 66, 0.7)');
    gradient.addColorStop(1, 'rgba(79, 121, 66, 0.0)');
    
    const optimalTrainingZoneChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: intensityRange.map(x => x + '%'),
            datasets: [
                {
                    label: 'Performance Benefit',
                    data: performanceBenefit,
                    borderColor: chartColors.primary,
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Injury Risk',
                    data: injuryRisk,
                    borderColor: chartColors.danger,
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Fatigue',
                    data: fatigueAccumulation,
                    borderColor: chartColors.info,
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Overall Benefit',
                    data: benefitScore.map(x => Math.max(0, x)),
                    borderColor: 'rgba(79, 121, 66, 1)',
                    backgroundColor: gradient,
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            ...globalChartOptions,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Training Intensity',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        maxTicksLimit: 11,
                        font: {
                            family: "'Poppins', sans-serif"
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    min: 0,
                    max: 100,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Value (%)',
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
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y1: {
                    min: 0,
                    max: 100,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Overall Benefit',
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
            plugins: {
                ...globalChartOptions.plugins,
                annotation: {
                    annotations: {
                        optimalRange: {
                            type: 'box',
                            xMin: optimalMin + '%',
                            xMax: optimalMax + '%',
                            backgroundColor: 'rgba(79, 121, 66, 0.1)',
                            borderColor: 'rgba(79, 121, 66, 0.5)',
                            borderWidth: 1,
                            label: {
                                display: true,
                                content: 'Optimal Training Zone',
                                position: 'center',
                                font: {
                                    family: "'Poppins', sans-serif",
                                    size: 12
                                },
                                color: 'rgba(79, 121, 66, 0.8)'
                            }
                        }
                    }
                }
            }
        }
    });
    
    window.optimalTrainingZoneChart = optimalTrainingZoneChart;
    
    document.getElementById('optimalIntensityValue').textContent = `${optimalMin}-${optimalMax}%`;
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
    
    const downloadBtn = document.getElementById('downloadZoneChart');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            const canvas = document.getElementById('optimalTrainingZoneChart');
            const link = document.createElement('a');
            link.download = 'optimal-training-zone.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        });
    }
    
    const refreshBtn = document.getElementById('refreshZoneChart');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            refreshBtn.classList.add('rotating');
            
            setTimeout(function() {
                initOptimalTrainingZoneChart();
                refreshBtn.classList.remove('rotating');
            }, 1000);
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
function initCombinedMetricsChart() {
    const ctx = document.getElementById('combinedMetricsChart').getContext('2d');
    
    // Sample data for the last 14 days
    const dates = getLastNDays(14);
    // Training intensity using discrete values (0.3, 0.6, 0.9)
    const trainingIntensity = [0.3, 0.6, 0.9, 0.3, 0.3, 0.6, 0.9, 0.6, 0.3, 0.6, 0.6, 0.9, 0.3, 0.6];
    // Performance on scale 0-10
    const performance = [7.6, 7.0, 6.5, 7.2, 7.8, 7.4, 8.0, 8.2, 8.5, 8.0, 7.5, 6.8, 7.8, 8.2];
    // Risk as decimal (0-1)
    const injuryRisk = [0.15, 0.18, 0.28, 0.25, 0.20, 0.23, 0.30, 0.25, 0.15, 0.20, 0.30, 0.45, 0.30, 0.25];
    // Fatigue on scale 0-4
    const fatigue = [1.2, 1.8, 2.4, 2.2, 1.6, 2.0, 2.6, 2.3, 1.4, 1.8, 2.2, 2.8, 2.2, 1.8];
    
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
function initRiskHeatmap() {
    const heatmapContainer = document.getElementById('riskHeatmap');
    if (!heatmapContainer) return;
    
    // Clear existing content
    heatmapContainer.innerHTML = '';
    
    // Define weeks and days
    const weeks = 4; // 4 weeks of data
    const weekLabels = ['Week 1', 'Week 2', 'Week 3', 'Week 4'];
    const daysOfWeek = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    
    // Create table structure for better layout
    const table = document.createElement('table');
    table.className = 'heatmap-table';
    
    // Create header row with days
    const headerRow = document.createElement('tr');
    
    // Add empty cell for corner
    const cornerCell = document.createElement('th');
    headerRow.appendChild(cornerCell);
    
    // Add day headers
    daysOfWeek.forEach(day => {
        const th = document.createElement('th');
        th.textContent = day;
        headerRow.appendChild(th);
    });
    
    table.appendChild(headerRow);
    
    // Risk level ranges and colors
    const riskLevels = [
        { min: 0, max: 0.3, color: 'rgba(79, 121, 66, 0.2)' },    // Low risk - Light green
        { min: 0.3, max: 0.6, color: 'rgba(79, 121, 66, 0.4)' },  // Moderate risk - Medium green
        { min: 0.6, max: 0.8, color: 'rgba(79, 121, 66, 0.6)' },  // High risk - Dark green
        { min: 0.8, max: 1.0, color: 'rgba(79, 121, 66, 0.8)' }   // Critical risk - Very dark green
    ];
    
    // Create data rows for each week
    for (let week = 0; week < weeks; week++) {
        const row = document.createElement('tr');
        
        // Add week label
        const weekLabelCell = document.createElement('td');
        weekLabelCell.textContent = weekLabels[week];
        row.appendChild(weekLabelCell);
        
        // Add day cells for this week
        for (let day = 0; day < 7; day++) {
            const cell = document.createElement('td');
            
            // Generate random risk value between 0 and 1
            const risk = Math.random();
            
            // Find the appropriate risk level and color
            const riskLevel = riskLevels.find(level => risk >= level.min && risk < level.max);
            const bgColor = riskLevel ? riskLevel.color : 'rgba(79, 121, 66, 0.2)';
            
            cell.style.backgroundColor = bgColor;
            
            // Add tooltip with risk info
            cell.title = `Risk Level: ${risk.toFixed(2)}`;
            
            row.appendChild(cell);
        }
        
        table.appendChild(row);
    }
    
    heatmapContainer.appendChild(table);
}

// Initialize Metrics Table
function initMetricsTable() {
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
    
    // Generate sample data for the table
    const days = 14; // Last 14 days
    const dates = getLastNDays(days);
    
    // Standard intensity values
    const intensityValues = [0.3, 0.6, 0.9];
    
    for (let i = 0; i < days; i++) {
        // Generate data using our standardized scales
        const intensity = intensityValues[Math.floor(Math.random() * intensityValues.length)];
        const performance = (Math.random() * 3 + 7).toFixed(1); // Performance between 7-10
        const risk = (Math.random() * 0.4).toFixed(2); // Risk between 0-0.4
        const fatigue = (Math.random() * 2.5 + 1).toFixed(1); // Fatigue between 1-3.5
        
        // Create table row
        const row = document.createElement('tr');
        
        // Create cells
        row.innerHTML = `
            <td>${dates[i]}</td>
            <td>${intensity}</td>
            <td>${performance}</td>
            <td>${risk}</td>
            <td>${fatigue}</td>
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
        
        // Add row to table
        tableBody.appendChild(row);
    }
    
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