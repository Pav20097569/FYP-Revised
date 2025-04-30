"""
Model Visualization Dashboard for Forza + CANoe Telemetry with improved path handling

This script creates a dashboard to visualize the model analysis results.
"""

from flask import Flask, render_template, jsonify, send_from_directory, send_file
import os
import json
import base64
import glob

def create_model_dashboard_app(port=5301):
    """Create a Flask app for visualizing model analysis results"""
    # Get absolute paths for better reliability
    analysis_dir = os.path.abspath("analysis_results")
    model_analysis_dir = os.path.join(analysis_dir, "model_analysis")
    
    # Verify directories exist
    print(f"Checking if analysis directory exists: {analysis_dir}")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir, exist_ok=True)
        print(f"Created missing directory: {analysis_dir}")
        
    print(f"Checking if model_analysis directory exists: {model_analysis_dir}")
    if not os.path.exists(model_analysis_dir):
        os.makedirs(model_analysis_dir, exist_ok=True)
        print(f"Created missing directory: {model_analysis_dir}")
    
    # List files in the model_analysis directory
    print("Files in model_analysis directory:")
    if os.path.exists(model_analysis_dir):
        files = os.listdir(model_analysis_dir)
        for file in files:
            print(f" - {file}")
    else:
        print(" - Directory not found")
    
    app = Flask(__name__, static_folder=analysis_dir)
    
    @app.route('/')
    def index():
        """Serve the dashboard HTML"""
        return render_template('model_dashboard.html')
    
    @app.route('/api/model-reports')
    def get_model_reports():
        """Get the model analysis reports"""
        reports_path = os.path.join(model_analysis_dir, "model_reports.json")
        print(f"Looking for model reports at: {reports_path}")
        
        if os.path.exists(reports_path):
            try:
                with open(reports_path, 'r') as f:
                    return jsonify(json.load(f))
            except Exception as e:
                print(f"Error loading model reports: {e}")
                return jsonify({"error": f"Could not load model reports: {e}"})
        else:
            print(f"Model reports file not found at {reports_path}")
            return jsonify({"error": "No model reports found"})
    
    @app.route('/api/anomaly-report')
    def get_anomaly_report():
        """Get the anomaly detector report"""
        report_path = os.path.join(model_analysis_dir, "anomaly_detector_report.json")
        print(f"Looking for anomaly report at: {report_path}")
        
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    return jsonify(json.load(f))
            except Exception as e:
                print(f"Error loading anomaly report: {e}")
                return jsonify({"error": f"Could not load anomaly report: {e}"})
        else:
            print(f"Anomaly report file not found at {report_path}")
            return jsonify({"error": "No anomaly report found"})
    
    @app.route('/api/plots')
    def get_plots():
        """Get all available plot images"""
        plot_files = glob.glob(os.path.join(model_analysis_dir, "*.png"))
        print(f"Found {len(plot_files)} plot files")
        plots = []
        
        for plot_file in plot_files:
            plot_name = os.path.basename(plot_file)
            # Categorize plots
            if "feature_importance" in plot_name:
                category = "feature_importance"
                title = "Feature Importance: " + plot_name.split("_feature_importance")[0]
            elif "roc_curve" in plot_name:
                category = "roc_curve"
                title = "ROC Curve: " + plot_name.split("_roc_curve")[0]
            elif "anomaly_score_distribution" in plot_name:
                category = "anomaly"
                title = "Anomaly Score Distribution"
            elif "anomaly_" in plot_name and "_distribution" in plot_name:
                category = "anomaly"
                feature = plot_name.split("anomaly_")[1].split("_distribution")[0]
                title = f"Distribution for {feature}: Normal vs Anomalies"
            else:
                category = "other"
                title = plot_name
            
            plots.append({
                "filename": plot_name,
                "path": f"/model_analysis/{plot_name}",
                "category": category,
                "title": title
            })
        
        return jsonify(plots)
    
    # Add a direct file serving route for debugging
    @app.route('/image/<filename>')
    def serve_image(filename):
        """Directly serve an image file for debugging"""
        file_path = os.path.join(model_analysis_dir, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return f"File not found: {file_path}", 404
    
    # Direct route to access model_analysis files
    @app.route('/model_analysis/<path:filename>')
    def serve_model_analysis(filename):
        """Serve files from the model_analysis directory"""
        return send_from_directory(model_analysis_dir, filename)
    
    # Ensure template directory exists
    template_dir = "templates"
    os.makedirs(template_dir, exist_ok=True)
    
    # Create HTML template with fixed image paths
    dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>Model Analysis Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #121212;
            color: #ffffff;
            margin: 0;
            padding: 0;
        }
        
        .navbar {
            background: #1e1e1e;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .navbar h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 0 20px;
        }
        
        .card {
            background: #1e1e1e;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        
        .card-title i {
            margin-right: 10px;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 15px;
            border-bottom: 1px solid #333;
        }
        
        .tab {
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
        }
        
        .tab.active {
            border-bottom-color: #2196F3;
            color: #2196F3;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }
        
        .plot-item {
            background: #252525;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .plot-title {
            padding: 10px;
            background: #333;
            font-weight: bold;
            text-align: center;
        }
        
        .plot-image {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .model-details {
            margin-top: 10px;
        }
        
        .feature-importance {
            margin-top: 15px;
        }
        
        .feature-bar {
            height: 20px;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
        }
        
        .feature-fill {
            background: #2196F3;
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 3px;
        }
        
        .feature-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
        }
        
        .feature-name {
            margin-right: 10px;
        }
        
        .feature-value {
            color: #aaa;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat-item {
            background: #252525;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            margin: 5px 0;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #aaa;
        }
        
        .error-message {
            color: #ff6b6b;
            text-align: center;
            padding: 20px;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            font-size: 1.2rem;
            color: #aaa;
        }
        
        /* Debug section for file paths */
        .debug-info {
            background: #333;
            padding: 10px;
            margin-top: 20px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h1><i class="fas fa-chart-line"></i> Model Analysis Dashboard</h1>
    </div>
    
    <div class="container">
        <div class="card">
            <div class="tabs">
                <div class="tab active" data-tab="dtc-models">DTC Prediction Models</div>
                <div class="tab" data-tab="anomaly-detector">Anomaly Detector</div>
                <div class="tab" data-tab="visualizations">Visualizations</div>
                <div class="tab" data-tab="debug">Debug</div>
            </div>
            
            <!-- DTC Models Tab -->
            <div class="tab-content active" id="dtc-models">
                <div id="dtc-models-loading" class="loading">
                    <i class="fas fa-spinner fa-spin"></i> Loading DTC model data...
                </div>
                <div id="dtc-models-content"></div>
            </div>
            
            <!-- Anomaly Detector Tab -->
            <div class="tab-content" id="anomaly-detector">
                <div id="anomaly-loading" class="loading">
                    <i class="fas fa-spinner fa-spin"></i> Loading anomaly detector data...
                </div>
                <div id="anomaly-content"></div>
            </div>
            
            <!-- Visualizations Tab -->
            <div class="tab-content" id="visualizations">
                <div id="visualizations-loading" class="loading">
                    <i class="fas fa-spinner fa-spin"></i> Loading visualizations...
                </div>
                <div id="visualizations-content"></div>
            </div>
            
            <!-- Debug Tab -->
            <div class="tab-content" id="debug">
                <h3>Image Path Testing</h3>
                <p>Test if images can be loaded directly:</p>
                <div id="debug-images">
                    <!-- Will be filled by JavaScript -->
                </div>
                
                <h3>API Responses</h3>
                <div class="debug-info" id="debug-api-plots"></div>
                <div class="debug-info" id="debug-api-models"></div>
                <div class="debug-info" id="debug-api-anomaly"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                
                // Add active class to clicked tab
                tab.classList.add('active');
                
                // Hide all tab content
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // Show related tab content
                const tabContentId = tab.getAttribute('data-tab');
                document.getElementById(tabContentId).classList.add('active');
                
                // Load debug info if debug tab is selected
                if (tabContentId === 'debug') {
                    loadDebugInfo();
                }
            });
        });
        
        // Function to load debug information
        function loadDebugInfo() {
            // Test some common image files
            const testImages = [
                'anomaly_score_distribution.png',
                'anomaly_throttle_distribution.png', 
                'anomaly_rpm_distribution.png',
                'dtc_P0300_feature_importance.png'
            ];
            
            const debugImagesDiv = document.getElementById('debug-images');
            debugImagesDiv.innerHTML = '';
            
            testImages.forEach(img => {
                const directPath = `/image/${img}`;
                const normalPath = `/model_analysis/${img}`;
                
                const testElement = document.createElement('div');
                testElement.style.marginBottom = '20px';
                testElement.innerHTML = `
                    <p><strong>Testing image:</strong> ${img}</p>
                    <p>Direct path: <a href="${directPath}" target="_blank">${directPath}</a></p>
                    <p>Normal path: <a href="${normalPath}" target="_blank">${normalPath}</a></p>
                    <div style="display: flex; gap: 10px;">
                        <div>
                            <p>Direct:</p>
                            <img src="${directPath}" alt="${img}" style="max-width: 200px; border: 1px solid #666;">
                        </div>
                        <div>
                            <p>Normal:</p>
                            <img src="${normalPath}" alt="${img}" style="max-width: 200px; border: 1px solid #666;">
                        </div>
                    </div>
                `;
                debugImagesDiv.appendChild(testElement);
            });
            
            // Get API responses
            fetch('/api/plots')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('debug-api-plots').textContent = 
                        'API /api/plots response: ' + JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    document.getElementById('debug-api-plots').textContent = 
                        'Error fetching plots: ' + error.message;
                });
                
            fetch('/api/model-reports')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('debug-api-models').textContent = 
                        'API /api/model-reports response: ' + JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    document.getElementById('debug-api-models').textContent = 
                        'Error fetching model reports: ' + error.message;
                });
                
            fetch('/api/anomaly-report')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('debug-api-anomaly').textContent = 
                        'API /api/anomaly-report response: ' + JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    document.getElementById('debug-api-anomaly').textContent = 
                        'Error fetching anomaly report: ' + error.message;
                });
        }
        
        // Load DTC model data
        fetch('/api/model-reports')
            .then(response => response.json())
            .then(data => {
                document.getElementById('dtc-models-loading').style.display = 'none';
                const contentElement = document.getElementById('dtc-models-content');
                
                if (data.error) {
                    contentElement.innerHTML = `<div class="error-message">${data.error}</div>`;
                    return;
                }
                
                // Create content for each DTC model
                const dtcKeys = Object.keys(data);
                if (dtcKeys.length === 0) {
                    contentElement.innerHTML = '<div class="error-message">No DTC models found</div>';
                    return;
                }
                
                dtcKeys.forEach(dtcCode => {
                    const modelInfo = data[dtcCode];
                    const modelElement = document.createElement('div');
                    modelElement.className = 'model-details';
                    
                    // Create model header
                    const header = document.createElement('h3');
                    header.textContent = `Model for ${dtcCode}`;
                    modelElement.appendChild(header);
                    
                    // Model parameters
                    const params = modelInfo.model_params;
                    if (params) {
                        const paramsElement = document.createElement('div');
                        paramsElement.innerHTML = `
                            <p>Model Parameters:</p>
                            <ul>
                                <li>Number of Trees: ${params.n_estimators || 'N/A'}</li>
                                <li>Max Depth: ${params.max_depth || 'None (unlimited)'}</li>
                                <li>Class Weight: ${params.class_weight || 'None'}</li>
                            </ul>
                        `;
                        modelElement.appendChild(paramsElement);
                    }
                    
                    // Feature importance
                    const importance = modelInfo.feature_importance;
                    if (importance) {
                        const featureImportanceElement = document.createElement('div');
                        featureImportanceElement.className = 'feature-importance';
                        featureImportanceElement.innerHTML = '<p>Feature Importance:</p>';
                        
                        // Sort features by importance
                        const features = Object.keys(importance).sort((a, b) => importance[b] - importance[a]);
                        
                        // Check if all importance values are 0
                        const allZeros = features.every(f => importance[f] === 0);
                        
                        if (allZeros) {
                            featureImportanceElement.innerHTML += `
                                <div class="error-message">
                                    All feature importance values are 0. This suggests the model didn't find meaningful patterns in the data.
                                </div>
                            `;
                        } else {
                            // Show top 10 features
                            features.slice(0, 10).forEach(feature => {
                                const importanceValue = importance[feature];
                                const percentValue = (importanceValue * 100).toFixed(1);
                                
                                featureImportanceElement.innerHTML += `
                                    <div class="feature-label">
                                        <span class="feature-name">${feature}</span>
                                        <span class="feature-value">${percentValue}%</span>
                                    </div>
                                    <div class="feature-bar">
                                        <div class="feature-fill" style="width: ${percentValue}%;"></div>
                                    </div>
                                `;
                            });
                        }
                        
                        modelElement.appendChild(featureImportanceElement);
                    }
                    
                    contentElement.appendChild(modelElement);
                    contentElement.appendChild(document.createElement('hr'));
                });
            })
            .catch(error => {
                document.getElementById('dtc-models-loading').style.display = 'none';
                document.getElementById('dtc-models-content').innerHTML = `
                    <div class="error-message">Error loading model data: ${error.message}</div>
                `;
            });
        
        // Load anomaly detector data
        fetch('/api/anomaly-report')
            .then(response => response.json())
            .then(data => {
                document.getElementById('anomaly-loading').style.display = 'none';
                const contentElement = document.getElementById('anomaly-content');
                
                if (data.error) {
                    contentElement.innerHTML = `<div class="error-message">${data.error}</div>`;
                    return;
                }
                
                // Create statistics grid
                const statsElement = document.createElement('div');
                statsElement.className = 'stats-grid';
                
                // Add anomaly statistics
                statsElement.innerHTML = `
                    <div class="stat-item">
                        <div class="stat-value">${data.total_samples ? data.total_samples.toLocaleString() : 'N/A'}</div>
                        <div class="stat-label">Total Samples</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.anomaly_count ? data.anomaly_count.toLocaleString() : 'N/A'}</div>
                        <div class="stat-label">Anomalies Detected</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.anomaly_rate ? data.anomaly_rate.toFixed(2) : 'N/A'}%</div>
                        <div class="stat-label">Anomaly Rate</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.normal_count ? data.normal_count.toLocaleString() : 'N/A'}</div>
                        <div class="stat-label">Normal Samples</div>
                    </div>
                `;
                
                contentElement.appendChild(statsElement);
                
                // Add feature statistics
                if (data.feature_stats) {
                    const featureStatsElement = document.createElement('div');
                    featureStatsElement.innerHTML = '<h3>Feature Statistics</h3>';
                    
                    // Create a table for each feature
                    Object.keys(data.feature_stats).forEach(feature => {
                        const featureData = data.feature_stats[feature];
                        
                        featureStatsElement.innerHTML += `
                            <div class="card" style="margin-top: 15px;">
                                <h4>${feature}</h4>
                                <table style="width: 100%; border-collapse: collapse;">
                                    <thead>
                                        <tr>
                                            <th style="text-align: left; padding: 8px; border-bottom: 1px solid #333;">Statistic</th>
                                            <th style="text-align: right; padding: 8px; border-bottom: 1px solid #333;">Normal</th>
                                            <th style="text-align: right; padding: 8px; border-bottom: 1px solid #333;">Anomaly</th>
                                            <th style="text-align: right; padding: 8px; border-bottom: 1px solid #333;">Difference</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td style="text-align: left; padding: 8px; border-bottom: 1px solid #222;">Mean</td>
                                            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">
                                                ${featureData.normal && featureData.normal.mean !== undefined ? featureData.normal.mean.toFixed(2) : 'N/A'}
                                            </td>
                                            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">
                                                ${featureData.anomaly && featureData.anomaly.mean !== undefined ? featureData.anomaly.mean.toFixed(2) : 'N/A'}
                                            </td>
                                            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">
                                                ${(featureData.normal && featureData.anomaly && 
                                                   featureData.normal.mean !== undefined && featureData.anomaly.mean !== undefined) 
                                                  ? (featureData.anomaly.mean - featureData.normal.mean).toFixed(2) : 'N/A'}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td style="text-align: left; padding: 8px; border-bottom: 1px solid #222;">Std Dev</td>
                                            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">
                                                ${featureData.normal && featureData.normal.std !== undefined ? featureData.normal.std.toFixed(2) : 'N/A'}
                                            </td>
                                            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">
                                                ${featureData.anomaly && featureData.anomaly.std !== undefined ? featureData.anomaly.std.toFixed(2) : 'N/A'}
                                            </td>
                                            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">
                                                ${(featureData.normal && featureData.anomaly && 
                                                   featureData.normal.std !== undefined && featureData.anomaly.std !== undefined) 
                                                  ? (featureData.anomaly.std - featureData.normal.std).toFixed(2) : 'N/A'}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td style="text-align: left; padding: 8px; border-bottom: 1px solid #222;">Min</td>
                                            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">
                                                ${featureData.normal && featureData.normal.min !== undefined ? featureData.normal.min.toFixed(2) : 'N/A'}
                                            </td>
                                            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">
                                                ${featureData.anomaly && featureData.anomaly.min !== undefined ? featureData.anomaly.min.toFixed(2) : 'N/A'}
                                            </td>
                                            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">
                                                ${(featureData.normal && featureData.anomaly && 
                                                   featureData.normal.min !== undefined && featureData.anomaly.min !== undefined) 
                                                  ? (featureData.anomaly.min - featureData.normal.min).toFixed(2) : 'N/A'}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td style="text-align: left; padding: 8px;">Max</td>
                                            <td style="text-align: right; padding: 8px;">
                                                ${featureData.normal && featureData.normal.max !== undefined ? featureData.normal.max.toFixed(2) : 'N/A'}
                                            </td>
                                            <td style="text-align: right; padding: 8px;">
                                                ${featureData.anomaly && featureData.anomaly.max !== undefined ? featureData.anomaly.max.toFixed(2) : 'N/A'}
                                            </td>
                                            <td style="text-align: right; padding: 8px;">
                                                ${(featureData.normal && featureData.anomaly && 
                                                   featureData.normal.max !== undefined && featureData.anomaly.max !== undefined) 
                                                  ? (featureData.anomaly.max - featureData.normal.max).toFixed(2) : 'N/A'}
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        `;
                    });
                    
                    contentElement.appendChild(featureStatsElement);
                }
            })
            .catch(error => {
                document.getElementById('anomaly-loading').style.display = 'none';
                document.getElementById('anomaly-content').innerHTML = `
                    <div class="error-message">Error loading anomaly data: ${error.message}</div>
                `;
            });
        
        // Load visualizations
        fetch('/api/plots')
            .then(response => response.json())
            .then(data => {
                document.getElementById('visualizations-loading').style.display = 'none';
                const contentElement = document.getElementById('visualizations-content');
                
                if (data.length === 0) {
                    contentElement.innerHTML = '<div class="error-message">No visualizations found</div>';
                    return;
                }
                
                // Group plots by category
                const categories = {};
                data.forEach(plot=> {
                    if (!categories[plot.category]) {
                        categories[plot.category] = [];
                    }
                    categories[plot.category].push(plot);
                });
                
                // Create sections for each category
                Object.keys(categories).forEach(category => {
                    const categoryTitle = document.createElement('h3');
                    categoryTitle.textContent = category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    contentElement.appendChild(categoryTitle);
                    
                    const plotGrid = document.createElement('div');
                    plotGrid.className = 'plot-grid';
                    
                    categories[category].forEach(plot => {
                        const plotElement = document.createElement('div');
                        plotElement.className = 'plot-item';
                        
                        plotElement.innerHTML = `
                            <div class="plot-title">${plot.title}</div>
                            <img src="${plot.path}" class="plot-image" alt="${plot.title}" onerror="this.onerror=null; this.src='/image/${plot.filename}'; this.style.border='2px solid orange'; this.title='Using fallback path';">
                        `;
                        
                        plotGrid.appendChild(plotElement);
                    });
                    
                    contentElement.appendChild(plotGrid);
                });
            })
            .catch(error => {
                document.getElementById('visualizations-loading').style.display = 'none';
                document.getElementById('visualizations-content').innerHTML = `
                    <div class="error-message">Error loading visualizations: ${error.message}</div>
                `;
            });
    </script>
</body>
</html>"""
    
    # Write the HTML to the templates directory
    with open(os.path.join(template_dir, "model_dashboard.html"), 'w') as f:
        f.write(dashboard_html)
    
    print(f"Model dashboard will be available at http://localhost:{port}")
    print(f"If images don't load, check the Debug tab for detailed information")
    app.run(host="0.0.0.0", port=port, debug=True)
    
if __name__ == "__main__":
    create_model_dashboard_app()