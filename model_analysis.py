"""
Model Analysis Utility

This script provides tools to analyze the trained machine learning models
for DTC prediction and anomaly detection.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance

class ModelAnalyzer:
    def __init__(self, models_dir="models", results_dir="analysis_results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.dtc_models = None
        self.anomaly_detector = None
        self.reports = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "model_analysis"), exist_ok=True)
    
    def load_models(self):
        """Load all trained models"""
        # Load DTC prediction models
        dtc_model_path = os.path.join(self.models_dir, 'dtc_prediction_models.joblib')
        if os.path.exists(dtc_model_path):
            try:
                self.dtc_models = joblib.load(dtc_model_path)
                print(f"Loaded {len(self.dtc_models)} DTC prediction models")
            except Exception as e:
                print(f"Error loading DTC models: {e}")
        
        # Load anomaly detector
        anomaly_model_path = os.path.join(self.models_dir, 'telemetry_anomaly_detector.joblib')
        if os.path.exists(anomaly_model_path):
            try:
                self.anomaly_detector = joblib.load(anomaly_model_path)
                print(f"Loaded anomaly detector model")
            except Exception as e:
                print(f"Error loading anomaly detector: {e}")
        
        return self.dtc_models is not None or self.anomaly_detector is not None
    
    def analyze_dtc_models(self, telemetry_data=None):
        """Analyze DTC prediction models"""
        if self.dtc_models is None:
            print("No DTC models loaded")
            return False
        
        # If telemetry data is not provided, try to load from exports
        if telemetry_data is None:
            telemetry_data = self._load_telemetry_data()
            if telemetry_data is None:
                print("No telemetry data available for analysis")
                return False
        
        for dtc_code, model_info in self.dtc_models.items():
            print(f"Analyzing model for {dtc_code}...")
            
            # Extract model components
            model = model_info['model']
            features = model_info['features']
            
            # Get feature importance
            importances = model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]
            
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importance for {dtc_code}")
            plt.bar(range(len(importances)), importances[indices], yerr=std[indices], align="center")
            plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
            plt.xlim([-1, min(10, len(importances))])  # Show top 10 features max
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.results_dir, "model_analysis", f"{dtc_code}_feature_importance.png")
            plt.savefig(output_path)
            plt.close()
            
            # Calculate ROC curve if we have predictions from test set
            if hasattr(model, "_test_data") and hasattr(model, "_test_predictions"):
                test_data = model._test_data
                test_predictions = model._test_predictions
                
                # ROC curve
                fpr, tpr, _ = roc_curve(test_data['y_true'], test_data['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve for {dtc_code}')
                plt.legend(loc="lower right")
                plt.tight_layout()
                
                # Save ROC curve
                output_path = os.path.join(self.results_dir, "model_analysis", f"{dtc_code}_roc_curve.png")
                plt.savefig(output_path)
                plt.close()
            
            # Store analysis results
            self.reports[dtc_code] = {
                'feature_importance': {features[i]: float(importances[i]) for i in range(len(features))},
                'model_params': {
                    'n_estimators': model.n_estimators,
                    'max_depth': model.max_depth if hasattr(model, 'max_depth') else None,
                    'class_weight': str(model.class_weight) if hasattr(model, 'class_weight') else None
                }
            }
        
        # Save reports to JSON
        with open(os.path.join(self.results_dir, "model_analysis", "model_reports.json"), 'w') as f:
            json.dump(self.reports, f, indent=2)
        
        return True
    
    def analyze_anomaly_detector(self, telemetry_data=None):
        """Analyze anomaly detection model"""
        if self.anomaly_detector is None or 'model' not in self.anomaly_detector:
            print("No anomaly detector model loaded")
            return False
        
        # If telemetry data is not provided, try to load from exports
        if telemetry_data is None:
            telemetry_data = self._load_telemetry_data()
            if telemetry_data is None:
                print("No telemetry data available for analysis")
                return False
        
        # Extract model components
        model = self.anomaly_detector['model']
        scaler = self.anomaly_detector['scaler']
        features = self.anomaly_detector['features']
        
        # Prepare features
        X = telemetry_data[features].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Get anomaly scores
        anomaly_scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)
        
        # Distribution of anomaly scores
        plt.figure(figsize=(10, 6))
        plt.hist(anomaly_scores, bins=50)
        plt.axvline(x=0, color='r', linestyle='--', label='Threshold')
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        
        # Save the histogram
        output_path = os.path.join(self.results_dir, "model_analysis", "anomaly_score_distribution.png")
        plt.savefig(output_path)
        plt.close()
        
        # Sample anomalies analysis - compare feature distributions
        anomalies = telemetry_data[predictions == -1]
        normal = telemetry_data[predictions == 1]
        
        # Create feature distribution plots for anomalies vs normal data
        for feature in features:
            plt.figure(figsize=(10, 6))
            plt.hist(normal[feature], bins=30, alpha=0.5, label='Normal', density=True)
            
            if len(anomalies) > 0:
                plt.hist(anomalies[feature], bins=30, alpha=0.5, label='Anomalies', density=True)
            
            plt.title(f'{feature} Distribution - Normal vs Anomalies')
            plt.xlabel(feature)
            plt.ylabel('Normalized Frequency')
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            output_path = os.path.join(self.results_dir, "model_analysis", f"anomaly_{feature}_distribution.png")
            plt.savefig(output_path)
            plt.close()
        
        # Anomaly detection statistics
        anomaly_rate = (predictions == -1).mean() * 100
        
        # Create report
        anomaly_report = {
            'anomaly_rate': float(anomaly_rate),
            'total_samples': len(predictions),
            'anomaly_count': int((predictions == -1).sum()),
            'normal_count': int((predictions == 1).sum()),
            'feature_stats': {}
        }
        
        # Add feature statistics for normal vs anomalous data
        for feature in features:
            if len(anomalies) > 0:
                anomaly_report['feature_stats'][feature] = {
                    'normal': {
                        'mean': float(normal[feature].mean()),
                        'std': float(normal[feature].std()),
                        'min': float(normal[feature].min()),
                        'max': float(normal[feature].max())
                    },
                    'anomaly': {
                        'mean': float(anomalies[feature].mean()) if len(anomalies) > 0 else None,
                        'std': float(anomalies[feature].std()) if len(anomalies) > 0 else None,
                        'min': float(anomalies[feature].min()) if len(anomalies) > 0 else None,
                        'max': float(anomalies[feature].max()) if len(anomalies) > 0 else None
                    }
                }
        
        # Save report to JSON
        with open(os.path.join(self.results_dir, "model_analysis", "anomaly_detector_report.json"), 'w') as f:
            json.dump(anomaly_report, f, indent=2)
        
        return True
    
    def _load_telemetry_data(self):
        """Load telemetry data from exports"""
        # Look for telemetry exports
        export_dir = "telemetry_exports"
        if not os.path.exists(export_dir):
            return None
        
        csv_files = [os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith('.csv')]
        if not csv_files:
            return None
        
        # Load the most recent file
        latest_file = max(csv_files, key=os.path.getmtime)
        try:
            data = pd.read_csv(latest_file)
            print(f"Loaded {len(data)} records from {os.path.basename(latest_file)}")
            return data
        except Exception as e:
            print(f"Error loading telemetry data: {e}")
            return None

if __name__ == "__main__":
    # Create analyzer and run analysis
    analyzer = ModelAnalyzer()
    if analyzer.load_models():
        analyzer.analyze_dtc_models()
        analyzer.analyze_anomaly_detector()
        print("Analysis complete! Check the 'analysis_results/model_analysis' directory for reports.")
    else:
        print("No models found. Train models first before analyzing.")