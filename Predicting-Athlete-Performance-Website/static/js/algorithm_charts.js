// Algorithm Comparison Charts
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all charts
    initExecutionTimeChart();
    initMemoryUsageChart();
    initNodesExpandedChart();
    initPerformanceProgressionChart();
    initFatigueProgressionChart();
    initRiskProgressionChart();
    initTrainingIntensityChart();
    initTrainingDurationChart();
    initTrainingHeatmapChart();
    initRestDaysChart();
    initHighIntensityDaysChart();
    initTotalWorkloadChart();
    initAlgorithmRadarChart();
});

// Chart color palette
const algorithmColors = {
    CSP: '#9C27B0',
    DFS: '#F44336',
    BFS: '#2196F3',
    UCS: '#FF9800',
    Greedy: '#673AB7',
    AStar: '#4F7942',
    Genetic: '#E91E63'
};

// Common chart options
const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    aspectRatio: 1.5,  // More compact aspect ratio
    plugins: {
        legend: {
            position: 'bottom',
            labels: {
                font: {
                    family: "'Poppins', sans-serif",
                    size: 11  // Reduced font size
                },
                usePointStyle: true,
                boxWidth: 8   // Smaller legend items
            }
        },
        tooltip: {
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            titleColor: '#333',
            bodyColor: '#666',
            borderColor: '#ddd',
            borderWidth: 1,
            cornerRadius: 8,
            boxPadding: 5,
            usePointStyle: true,
            titleFont: {
                family: "'Poppins', sans-serif",
                size: 12,    // Reduced font size
                weight: 600
            },
            bodyFont: {
                family: "'Poppins', sans-serif",
                size: 11     // Reduced font size
            },
            padding: 8
        }
    }
};

// Initialize Execution Time Chart
function initExecutionTimeChart() {
    const ctx = document.getElementById('executionTimeChart').getContext('2d');
    
    // Placeholder data - execution times in seconds
    const data = {
        labels: ['CSP', 'DFS', 'BFS', 'UCS', 'Greedy', 'A*', 'Genetic'],
        datasets: [{
            label: 'Execution Time (seconds)',
            data: [3.8, 0.3, 5.4, 3.2, 1.2, 2.6, 4.2],
            backgroundColor: [
                algorithmColors.CSP,
                algorithmColors.DFS,
                algorithmColors.BFS,
                algorithmColors.UCS,
                algorithmColors.Greedy,
                algorithmColors.AStar,
                algorithmColors.Genetic
            ],
            borderColor: 'rgba(255, 255, 255, 0.7)',
            borderWidth: 2,
            borderRadius: 5
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (seconds)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins,
                title: {
                    display: false,
                    text: 'Execution Time Comparison',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        }
    });
}

// Initialize Memory Usage Chart
function initMemoryUsageChart() {
    const ctx = document.getElementById('memoryUsageChart').getContext('2d');
    
    // Placeholder data - memory usage in MB
    const data = {
        labels: ['CSP', 'DFS', 'BFS', 'UCS', 'Greedy', 'A*', 'Genetic'],
        datasets: [{
            label: 'Memory Usage (MB)',
            data: [1500, 320, 2200, 1800, 850, 1650, 780],
            backgroundColor: [
                algorithmColors.CSP,
                algorithmColors.DFS,
                algorithmColors.BFS,
                algorithmColors.UCS,
                algorithmColors.Greedy,
                algorithmColors.AStar,
                algorithmColors.Genetic
            ],
            borderColor: 'rgba(255, 255, 255, 0.7)',
            borderWidth: 2,
            borderRadius: 5
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Memory (MB)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins,
                title: {
                    display: false,
                    text: 'Memory Usage Comparison',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        }
    });
}

// Initialize Nodes Expanded Chart
function initNodesExpandedChart() {
    const ctx = document.getElementById('nodesExpandedChart').getContext('2d');
    
    // Placeholder data - nodes expanded
    const data = {
        labels: ['CSP', 'DFS', 'BFS', 'UCS', 'Greedy', 'A*', 'Genetic'],
        datasets: [{
            label: 'Nodes Expanded',
            data: [25000, 12000, 85000, 62000, 8500, 15000, 32000],
            backgroundColor: [
                algorithmColors.CSP,
                algorithmColors.DFS,
                algorithmColors.BFS,
                algorithmColors.UCS,
                algorithmColors.Greedy,
                algorithmColors.AStar,
                algorithmColors.Genetic
            ],
            borderColor: 'rgba(255, 255, 255, 0.7)',
            borderWidth: 2,
            borderRadius: 5
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Number of Nodes (log scale)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins,
                title: {
                    display: false,
                    text: 'Nodes Expanded Comparison',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        }
    });
}

// Initialize Performance Progression Chart
function initPerformanceProgressionChart() {
    const ctx = document.getElementById('performanceProgressionChart').getContext('2d');
    
    // Days for the X-axis
    const days = Array.from({length: 11}, (_, i) => `Day ${i}`);
    
    // Placeholder data - performance progression over 10 days
    const data = {
        labels: days,
        datasets: [
            {
                label: 'CSP',
                data: [5.5, 5.8, 6.1, 6.5, 6.9, 7.0, 7.2, 7.5, 7.7, 7.8, 8.0],
                borderColor: algorithmColors.CSP,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            },
            {
                label: 'DFS',
                data: [5.5, 5.7, 5.9, 6.2, 6.4, 6.5, 6.7, 6.8, 6.9, 7.0, 7.1],
                borderColor: algorithmColors.DFS,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            },
            {
                label: 'BFS',
                data: [5.5, 5.6, 5.9, 6.2, 6.5, 6.8, 7.0, 7.2, 7.4, 7.5, 7.7],
                borderColor: algorithmColors.BFS,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            },
            {
                label: 'A*',
                data: [5.5, 5.9, 6.3, 6.7, 7.0, 7.2, 7.5, 7.7, 7.9, 8.1, 8.2],
                borderColor: algorithmColors.AStar,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            }
        ]
    };
    
    new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 5,
                    max: 9,
                    title: {
                        display: true,
                        text: 'Performance',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins,
                annotation: {
                    annotations: {
                        targetLine: {
                            type: 'line',
                            yMin: 7.5,
                            yMax: 7.5,
                            borderColor: '#FF9800',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {
                                content: 'Target',
                                enabled: true,
                                position: 'end'
                            }
                        }
                    }
                }
            }
        }
    });
}

// Initialize Fatigue Progression Chart
function initFatigueProgressionChart() {
    const ctx = document.getElementById('fatigueProgressionChart').getContext('2d');
    
    // Days for the X-axis
    const days = Array.from({length: 11}, (_, i) => `Day ${i}`);
    
    // Placeholder data - fatigue progression over 10 days
    const data = {
        labels: days,
        datasets: [
            {
                label: 'CSP',
                data: [1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.7, 2.5, 2.3, 2.5, 2.7],
                borderColor: algorithmColors.CSP,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            },
            {
                label: 'DFS',
                data: [1.5, 1.9, 2.3, 2.7, 3.0, 2.6, 2.9, 3.2, 2.8, 3.1, 3.4],
                borderColor: algorithmColors.DFS,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            },
            {
                label: 'BFS',
                data: [1.5, 1.7, 1.9, 2.2, 2.5, 2.8, 2.6, 2.4, 2.7, 2.9, 2.7],
                borderColor: algorithmColors.BFS,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            },
            {
                label: 'A*',
                data: [1.5, 1.7, 1.9, 2.2, 2.5, 2.3, 2.1, 2.4, 2.6, 2.3, 2.1],
                borderColor: algorithmColors.AStar,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            }
        ]
    };
    
    new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 1,
                    max: 4,
                    title: {
                        display: true,
                        text: 'Fatigue',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins,
                annotation: {
                    annotations: {
                        maxFatigueLine: {
                            type: 'line',
                            yMin: 3.0,
                            yMax: 3.0,
                            borderColor: '#F44336',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {
                                content: 'Max',
                                enabled: true,
                                position: 'end'
                            }
                        }
                    }
                }
            }
        }
    });
}

// Initialize Risk Progression Chart
function initRiskProgressionChart() {
    const ctx = document.getElementById('riskProgressionChart').getContext('2d');
    
    // Days for the X-axis
    const days = Array.from({length: 11}, (_, i) => `Day ${i}`);
    
    // Placeholder data - risk progression over 10 days
    const data = {
        labels: days,
        datasets: [
            {
                label: 'CSP',
                data: [0.20, 0.22, 0.24, 0.27, 0.29, 0.31, 0.30, 0.28, 0.31, 0.33, 0.34],
                borderColor: algorithmColors.CSP,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            },
            {
                label: 'DFS',
                data: [0.20, 0.24, 0.28, 0.32, 0.35, 0.32, 0.36, 0.39, 0.36, 0.38, 0.40],
                borderColor: algorithmColors.DFS,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            },
            {
                label: 'BFS',
                data: [0.20, 0.22, 0.25, 0.28, 0.31, 0.33, 0.31, 0.29, 0.33, 0.35, 0.33],
                borderColor: algorithmColors.BFS,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            },
            {
                label: 'A*',
                data: [0.20, 0.22, 0.24, 0.26, 0.28, 0.27, 0.25, 0.27, 0.29, 0.28, 0.26],
                borderColor: algorithmColors.AStar,
                backgroundColor: 'transparent',
                tension: 0.3,
                borderWidth: 3
            }
        ]
    };
    
    new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0.15,
                    max: 0.45,
                    title: {
                        display: true,
                        text: 'Injury Risk',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins,
                annotation: {
                    annotations: {
                        maxRiskLine: {
                            type: 'line',
                            yMin: 0.35,
                            yMax: 0.35,
                            borderColor: '#F44336',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {
                                content: 'Max',
                                enabled: true,
                                position: 'end'
                            }
                        }
                    }
                }
            }
        }
    });
}

// Initialize Training Intensity Chart
function initTrainingIntensityChart() {
    const ctx = document.getElementById('trainingIntensityChart').getContext('2d');
    
    // Placeholder data - average training intensity for each day by algorithm
    const data = {
        labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', 'Day 10'],
        datasets: [
            {
                label: 'CSP',
                data: [0.6, 0.7, 0.8, 0.6, 0.5, 0.9, 0.7, 0.0, 0.8, 0.7],
                backgroundColor: algorithmColors.CSP
            },
            {
                label: 'A*',
                data: [0.7, 0.8, 0.6, 0.0, 0.8, 0.7, 0.5, 0.9, 0.0, 0.8],
                backgroundColor: algorithmColors.AStar
            },
            {
                label: 'DFS',
                data: [0.8, 0.9, 0.7, 0.8, 0.7, 0.8, 0.9, 0.7, 0.0, 0.8],
                backgroundColor: algorithmColors.DFS
            },
            {
                label: 'BFS',
                data: [0.7, 0.5, 0.0, 0.8, 0.9, 0.6, 0.7, 0.8, 0.5, 0.0],
                backgroundColor: algorithmColors.BFS
            }
        ]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                x: {
                    stacked: false,
                },
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Training Intensity',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins
            }
        }
    });
}

// Initialize Training Duration Chart
function initTrainingDurationChart() {
    const ctx = document.getElementById('trainingDurationChart').getContext('2d');
    
    // Placeholder data - average training duration for each day by algorithm
    const data = {
        labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', 'Day 10'],
        datasets: [
            {
                label: 'CSP',
                data: [45, 60, 75, 60, 45, 90, 60, 0, 90, 60],
                backgroundColor: algorithmColors.CSP
            },
            {
                label: 'A*',
                data: [60, 75, 60, 0, 90, 60, 45, 90, 0, 75],
                backgroundColor: algorithmColors.AStar
            },
            {
                label: 'DFS',
                data: [90, 90, 90, 90, 90, 90, 90, 60, 0, 90],
                backgroundColor: algorithmColors.DFS
            },
            {
                label: 'BFS',
                data: [60, 45, 0, 90, 90, 60, 75, 60, 45, 0],
                backgroundColor: algorithmColors.BFS
            }
        ]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                x: {
                    stacked: false,
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Duration (minutes)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins
            }
        }
    });
}

// Initialize Training Heatmap Chart
function initTrainingHeatmapChart() {
    const heatmapContainer = document.getElementById('trainingHeatmapChart').parentNode;
    if (!heatmapContainer) return;

    // Remove Canvas and create a div container for the GitHub-style heatmap
    const canvas = document.getElementById('trainingHeatmapChart');
    const containerDiv = document.createElement('div');
    containerDiv.id = 'algorithmHeatmapContainer';
    containerDiv.className = 'algorithm-heatmap-container';
    containerDiv.style.width = '100%';
    containerDiv.style.marginTop = '20px';
    canvas.parentNode.replaceChild(containerDiv, canvas);
    
    // Define algorithms and days
    const algorithms = ['CSP', 'DFS', 'BFS', 'UCS', 'Greedy', 'A*', 'Genetic'];
    const days = Array.from({length: 10}, (_, i) => `Day ${i+1}`);
    
    // Create heatmap heading row with days
    const headerRow = document.createElement('div');
    headerRow.className = 'heatmap-header-row';
    headerRow.style.display = 'grid';
    headerRow.style.gridTemplateColumns = '120px repeat(10, 1fr)';
    headerRow.style.gap = '5px';
    headerRow.style.marginBottom = '10px';
    headerRow.style.fontFamily = "'Poppins', sans-serif";
    headerRow.style.fontWeight = '600';
    headerRow.style.fontSize = '12px';
    
    // Add empty cell for corner
    const cornerCell = document.createElement('div');
    cornerCell.style.textAlign = 'right';
    cornerCell.style.paddingRight = '10px';
    cornerCell.textContent = 'Algorithm';
    headerRow.appendChild(cornerCell);
    
    // Add day headers
    days.forEach(day => {
        const dayHeader = document.createElement('div');
        dayHeader.textContent = day;
        dayHeader.style.textAlign = 'center';
        headerRow.appendChild(dayHeader);
    });
    
    containerDiv.appendChild(headerRow);
    
    // Generate intensity data for each algorithm and day
    const heatmapData = {};
    
    algorithms.forEach(algorithm => {
        heatmapData[algorithm] = [];
        for (let i = 0; i < 10; i++) {
            // Add algorithm-specific patterns
            if (algorithm === 'DFS' && i > 7) {
                // DFS tends to have high intensity throughout
                heatmapData[algorithm].push(Math.random() * 0.2 + 0.8);
            } else if (algorithm === 'BFS' && i % 2 === 0) {
                // BFS alternates between rest and high intensity
                heatmapData[algorithm].push(i % 4 === 0 ? 0 : Math.random() * 0.3 + 0.7);
            } else if (algorithm === 'A*' && (i === 3 || i === 7)) {
                // A* has strategic rest days
                heatmapData[algorithm].push(0);
            } else if (algorithm === 'CSP' && i % 3 === 2) {
                // CSP schedules rest every 3rd day
                heatmapData[algorithm].push(0);
            } else if (algorithm === 'Genetic' && i === 9) {
                // Genetic ends with rest
                heatmapData[algorithm].push(0);
            } else if (algorithm === 'Greedy' && (i === 0 || i === 5)) {
                // Greedy puts high intensity on certain days
                heatmapData[algorithm].push(0.9);
            } else if (algorithm === 'UCS' && i % 4 === 3) {
                // UCS has a pattern with rest every 4th day
                heatmapData[algorithm].push(0);
            } else {
                // For other days, generate semi-random values with some consistency
                const baseValue = {
                    'CSP': 0.65,
                    'DFS': 0.75,
                    'BFS': 0.6,
                    'UCS': 0.55,
                    'Greedy': 0.7,
                    'A*': 0.65,
                    'Genetic': 0.6
                }[algorithm] || 0.6;
                
                heatmapData[algorithm].push(
                    Math.max(0, Math.min(1, baseValue + (Math.random() * 0.4 - 0.2)))
                );
            }
        }
    });
    
    // Create a row for each algorithm
    algorithms.forEach(algorithm => {
        const row = document.createElement('div');
        row.className = 'heatmap-row';
        row.style.display = 'grid';
        row.style.gridTemplateColumns = '120px repeat(10, 1fr)';
        row.style.gap = '5px';
        row.style.marginBottom = '5px';
        
        // Add algorithm label
        const label = document.createElement('div');
        label.textContent = algorithm;
        label.style.textAlign = 'right';
        label.style.paddingRight = '10px';
        label.style.fontFamily = "'Poppins', sans-serif";
        label.style.fontWeight = '500';
        label.style.fontSize = '13px';
        row.appendChild(label);
        
        // Add cells for each day
        heatmapData[algorithm].forEach((intensity, dayIndex) => {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            cell.style.height = '25px';
            cell.style.borderRadius = '3px';
            cell.style.transition = 'transform 0.2s';
            cell.style.cursor = 'pointer';
            
            // Set color based on intensity
            let bgColor;
            if (intensity === 0) {
                // Rest day
                bgColor = '#ebedf0';
            } else if (intensity < 0.4) {
                bgColor = 'rgba(79, 121, 66, 0.2)';
            } else if (intensity < 0.7) {
                bgColor = 'rgba(79, 121, 66, 0.5)';
            } else if (intensity < 0.9) {
                bgColor = 'rgba(79, 121, 66, 0.8)';
            } else {
                bgColor = 'rgba(79, 121, 66, 1)';
            }
            
            cell.style.backgroundColor = bgColor;
            
            // Add tooltip
            const tooltip = intensity === 0 
                ? `Rest Day` 
                : `Training Intensity: ${Math.round(intensity * 100)}%`;
            cell.setAttribute('title', `${algorithm} - Day ${dayIndex + 1}: ${tooltip}`);
            
            // Add hover effect
            cell.addEventListener('mouseover', function() {
                this.style.transform = 'scale(1.1)';
            });
            cell.addEventListener('mouseout', function() {
                this.style.transform = 'scale(1)';
            });
            
            row.appendChild(cell);
        });
        
        containerDiv.appendChild(row);
    });

    // Add legend
    const legend = document.createElement('div');
    legend.className = 'heatmap-legend';
    legend.style.display = 'flex';
    legend.style.alignItems = 'center';
    legend.style.justifyContent = 'center';
    legend.style.marginTop = '15px';
    legend.style.fontFamily = "'Poppins', sans-serif";
    legend.style.fontSize = '12px';
    
    // Legend label
    const legendLabel = document.createElement('div');
    legendLabel.textContent = 'Training Intensity:';
    legendLabel.style.marginRight = '10px';
    legend.appendChild(legendLabel);
    
    // Legend items
    const intensityLevels = [
        { color: '#ebedf0', label: 'Rest Day' },
        { color: 'rgba(79, 121, 66, 0.2)', label: 'Low' },
        { color: 'rgba(79, 121, 66, 0.5)', label: 'Medium' },
        { color: 'rgba(79, 121, 66, 0.8)', label: 'High' },
        { color: 'rgba(79, 121, 66, 1)', label: 'Very High' }
    ];
    
    intensityLevels.forEach(level => {
        const item = document.createElement('div');
        item.style.display = 'flex';
        item.style.alignItems = 'center';
        item.style.marginRight = '12px';
        
        const colorSwatch = document.createElement('div');
        colorSwatch.style.width = '12px';
        colorSwatch.style.height = '12px';
        colorSwatch.style.backgroundColor = level.color;
        colorSwatch.style.borderRadius = '2px';
        colorSwatch.style.marginRight = '4px';
        
        const itemLabel = document.createElement('span');
        itemLabel.textContent = level.label;
        
        item.appendChild(colorSwatch);
        item.appendChild(itemLabel);
        legend.appendChild(item);
    });
    
    containerDiv.appendChild(legend);
    
    // Add CSS for print-friendly version
    const style = document.createElement('style');
    style.textContent = `
        @media print {
            .algorithm-heatmap-container {
                break-inside: avoid;
            }
        }
    `;
    document.head.appendChild(style);
}

// Initialize Rest Days Chart
function initRestDaysChart() {
    const ctx = document.getElementById('restDaysChart').getContext('2d');
    
    // Placeholder data - rest days for each algorithm
    const data = {
        labels: ['CSP', 'DFS', 'BFS', 'UCS', 'Greedy', 'A*', 'Genetic'],
        datasets: [{
            label: 'Rest Days',
            data: [2, 1, 2, 3, 1, 3, 2],
            backgroundColor: '#A5D6A7',
            borderColor: '#43A047',
            borderWidth: 2,
            borderRadius: 5
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 4,
                    title: {
                        display: true,
                        text: 'Number of Days',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins
            }
        }
    });
}

// Initialize High Intensity Days Chart
function initHighIntensityDaysChart() {
    const ctx = document.getElementById('highIntensityDaysChart').getContext('2d');
    
    // Placeholder data - high intensity days for each algorithm
    const data = {
        labels: ['CSP', 'DFS', 'BFS', 'UCS', 'Greedy', 'A*', 'Genetic'],
        datasets: [{
            label: 'High Intensity Days',
            data: [3, 5, 4, 3, 4, 2, 3],
            backgroundColor: '#FFCC80',
            borderColor: '#FB8C00',
            borderWidth: 2,
            borderRadius: 5
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 6,
                    title: {
                        display: true,
                        text: 'Number of Days',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins
            }
        }
    });
}

// Initialize Total Workload Chart
function initTotalWorkloadChart() {
    const ctx = document.getElementById('totalWorkloadChart').getContext('2d');
    
    // Placeholder data - total workload for each algorithm
    const data = {
        labels: ['CSP', 'DFS', 'BFS', 'UCS', 'Greedy', 'A*', 'Genetic'],
        datasets: [{
            label: 'Total Workload',
            data: [575, 650, 480, 520, 580, 540, 510],
            backgroundColor: '#81D4FA',
            borderColor: '#0288D1',
            borderWidth: 2,
            borderRadius: 5
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            ...commonChartOptions,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Workload Units',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                ...commonChartOptions.plugins
            }
        }
    });
}

// Initialize Algorithm Radar Chart
function initAlgorithmRadarChart() {
    const ctx = document.getElementById('algorithmRadarChart').getContext('2d');
    
    // Check if radar chart is available
    if (!Chart.controllers.radar) {
        console.warn("Radar chart type not available. Displaying fallback visualization.");
        
        // Create a fallback bar chart for the metrics
        const metrics = [
            'Performance Optimization', 
            'Low Fatigue', 
            'Low Risk', 
            'Execution Speed',
            'Low Memory Usage',
            'Rest Balance',
            'Workout Distribution'
        ];
        
        // Sample data for the algorithms
        const algorithmData = {
            'CSP': [8, 7, 6, 5, 4, 7, 8],
            'A*': [9, 8, 8, 6, 3, 9, 8],
            'DFS': [5, 3, 4, 9, 8, 4, 3],
            'Genetic': [8, 6, 7, 4, 6, 6, 7]
        };
        
        // Restructure data for a grouped bar chart as fallback
        const datasets = Object.keys(algorithmData).map(algorithm => {
            const colorKey = algorithm.replace('*', 'Star');
            return {
                label: algorithm,
                data: algorithmData[algorithm],
                backgroundColor: algorithmColors[colorKey] || '#4F7942'
            };
        });
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: metrics,
                datasets: datasets
            },
            options: {
                ...commonChartOptions,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 10,
                        title: {
                            display: true,
                            text: 'Score (0-10)',
                            font: { 
                                weight: 'bold',
                                size: 14
                            }
                        }
                    }
                },
                plugins: {
                    ...commonChartOptions.plugins,
                    title: {
                        display: true,
                        text: 'Algorithm Performance Metrics',
                        font: { size: 14 }
                    }
                }
            }
        });
        return;
    }
    
    // Placeholder data - radar chart metrics for each algorithm
    const data = {
        labels: [
            'Performance Optimization', 
            'Low Fatigue', 
            'Low Risk', 
            'Execution Speed',
            'Low Memory Usage',
            'Rest Balance',
            'Workout Distribution'
        ],
        datasets: [
            {
                label: 'CSP',
                data: [8, 7, 6, 5, 4, 7, 8],
                backgroundColor: 'rgba(156, 39, 176, 0.2)',
                borderColor: algorithmColors.CSP,
                borderWidth: 2,
                pointBackgroundColor: algorithmColors.CSP,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: algorithmColors.CSP
            },
            {
                label: 'A*',
                data: [9, 8, 8, 6, 3, 9, 8],
                backgroundColor: 'rgba(79, 121, 66, 0.2)',
                borderColor: algorithmColors.AStar,
                borderWidth: 2,
                pointBackgroundColor: algorithmColors.AStar,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: algorithmColors.AStar
            },
            {
                label: 'DFS',
                data: [5, 3, 4, 9, 8, 4, 3],
                backgroundColor: 'rgba(244, 67, 54, 0.2)',
                borderColor: algorithmColors.DFS,
                borderWidth: 2,
                pointBackgroundColor: algorithmColors.DFS,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: algorithmColors.DFS
            },
            {
                label: 'Genetic',
                data: [8, 6, 7, 4, 6, 6, 7],
                backgroundColor: 'rgba(233, 30, 99, 0.2)',
                borderColor: algorithmColors.Genetic,
                borderWidth: 2,
                pointBackgroundColor: algorithmColors.Genetic,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: algorithmColors.Genetic
            }
        ]
    };
    
    new Chart(ctx, {
        type: 'radar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 11
                        },
                        usePointStyle: true,
                        boxWidth: 8
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    titleColor: '#333',
                    bodyColor: '#666',
                    borderColor: '#ddd',
                    borderWidth: 1,
                    cornerRadius: 8,
                    boxPadding: 5
                }
            },
            elements: {
                line: {
                    tension: 0
                }
            },
            scales: {
                r: {
                    angleLines: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    beginAtZero: true,
                    min: 0,
                    max: 10,
                    ticks: {
                        stepSize: 2,
                        backdropColor: 'transparent',
                        color: 'rgba(0, 0, 0, 0.7)'
                    },
                    pointLabels: {
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}
