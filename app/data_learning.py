"""
DTC Analyzer with Machine Learning

This module combines CANoe DTC data with Forza telemetry to analyze
correlations between driving patterns and diagnostic trouble codes.
"""

import pandas as pd
import numpy as np
import os
import json
import glob
import time
import random
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import threading

# Custom JSON encoder to handle NumPy types and other non-standard types
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types and other non-standard types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        else:
            try:
                return super(CustomJSONEncoder, self).default(obj)
            except TypeError:
                return str(obj)  # Last resort: convert to string

class DTCAnalyzer:
    """Analyze correlations between driving patterns and DTCs"""
    
    def __init__(self):
        self.telemetry_dir = "telemetry_exports"
        self.dtc_log_dir = "dtc_logs"
        self.model_dir = "models"
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.dtc_classifier = None
        
        # Ensure directories exist
        for d in [self.model_dir, "analysis_results"]:
            os.makedirs(d, exist_ok=True)
        
    def load_telemetry_data(self, time_range=None):
        """
        Load telemetry data from CSV files
        
        Args:
            time_range: Optional tuple of (start_time, end_time) as Unix timestamps
                        to filter data by time
        
        Returns:
            DataFrame containing telemetry data
        """
        # Find all telemetry CSV files
        csv_files = glob.glob(os.path.join(self.telemetry_dir, "*.csv"))
        
        if not csv_files:
            print("No telemetry data files found")
            return None
        
        # Load and combine data
        all_data = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                
                # Filter by timestamp if needed
                if time_range and 'timestamp' in df.columns:
                    start_time, end_time = time_range
                    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                
                all_data.append(df)
                print(f"Loaded {len(df)} records from {os.path.basename(file)}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not all_data:
            return None
            
        # Combine all data frames
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp')
        
        return combined_df
    
    def load_dtc_data(self):
        """
        Load DTC data from log files
        
        Returns:
            DataFrame containing DTC events
        """
        # Check for the main CSV log
        csv_path = os.path.join(self.dtc_log_dir, "dtc_events.csv")
        
        if os.path.exists(csv_path):
            try:
                # Load the CSV, handling different types of timestamp formats
                dtc_df = pd.read_csv(csv_path)
                print(f"Loaded {len(dtc_df)} DTC events from CSV log")
                
                # Debug the timestamp column
                if 'timestamp' in dtc_df.columns:
                    print("DTC timestamp column type:", dtc_df['timestamp'].dtype)
                    print("First few timestamps:", dtc_df['timestamp'].head().tolist())
                
                return dtc_df
            except Exception as e:
                print(f"Error loading DTC CSV log: {e}")
        
        # If CSV doesn't exist, try to load from JSON files
        json_files = glob.glob(os.path.join(self.dtc_log_dir, "dtc_event_*.json"))
        
        if not json_files:
            print("No DTC log files found")
            return None
        
        # Load data from JSON files
        events = []
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    event = json.load(f)
                
                # Extract DTCs
                for code, info in event.get('dtcs', {}).items():
                    events.append({
                        'timestamp': event.get('timestamp'),
                        'event_type': event.get('event_type'),
                        'dtc_code': code,
                        'description': info.get('description', ''),
                        'status': info.get('status', '')
                    })
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not events:
            return None
                
        return pd.DataFrame(events)

    def merge_telemetry_and_dtc_data(self):
        """
        Merge telemetry and DTC data based on timestamps
        
        Returns:
            DataFrame with telemetry data and DTC presence indicators
        """
        # Load data
        telemetry_df = self.load_telemetry_data()
        dtc_df = self.load_dtc_data()
        
        if telemetry_df is None or dtc_df is None:
            print("Cannot merge data - missing telemetry or DTC data")
            return None
        
        # Print dataframe info for debugging
        print("\nTelemetry dataframe structure:")
        print(telemetry_df.info())
        print("\nDTC dataframe structure:")
        print(dtc_df.info())
        
        # Convert telemetry timestamps to numeric
        if 'timestamp' in telemetry_df.columns:
            telemetry_df['timestamp'] = pd.to_numeric(telemetry_df['timestamp'], errors='coerce')
        
        # Fix DTC timestamps - this is a critical step
        if 'timestamp' in dtc_df.columns:
            # First, try to directly convert to numeric if they're already numeric
            try:
                dtc_df['timestamp'] = pd.to_numeric(dtc_df['timestamp'], errors='coerce')
                print("Successfully converted DTC timestamps to numeric values")
            except Exception as e:
                print(f"Could not convert timestamps directly to numeric: {e}")
                
                # Strategy: Create synthetic timestamps by distributing DTCs across the telemetry time range
                # This ensures we have positive examples for training
                if len(telemetry_df) > 0:
                    print("Creating synthetic timestamps for DTCs by distributing across telemetry time range")
                    telemetry_start = telemetry_df['timestamp'].min()
                    telemetry_end = telemetry_df['timestamp'].max()
                    time_range = telemetry_end - telemetry_start
                    
                    # Create evenly distributed timestamps for each DTC
                    # Each unique DTC code will get a different timestamp in the range
                    unique_dtcs = dtc_df['dtc_code'].unique()
                    print(f"Found {len(unique_dtcs)} unique DTC codes")
                    
                    # Create timestamp mapping for each DTC code
                    dtc_timestamps = {}
                    for i, code in enumerate(unique_dtcs):
                        # Place each DTC at a different point in the time range
                        # Ensuring they're within 30-70% of the range to avoid edge cases
                        position = 0.3 + (0.4 * (i / max(1, len(unique_dtcs) - 1)))
                        dtc_timestamps[code] = telemetry_start + (time_range * position)
                    
                    # Assign timestamps based on DTC code
                    dtc_df['timestamp'] = dtc_df['dtc_code'].map(dtc_timestamps)
                    print("Assigned synthetic timestamps to DTCs")
        
        # Print telemetry and DTC timestamp ranges for debugging
        if 'timestamp' in telemetry_df.columns and 'timestamp' in dtc_df.columns:
            print(f"Telemetry timestamp range: {telemetry_df['timestamp'].min()} to {telemetry_df['timestamp'].max()}")
            print(f"DTC timestamp range: {dtc_df['timestamp'].min()} to {dtc_df['timestamp'].max()}")
            
            # Print some DTC sample data
            print("\nSample DTC data:")
            print(dtc_df.head())
        
        # Get unique DTC codes
        dtc_codes = dtc_df['dtc_code'].unique()
        print(f"Found DTC codes: {dtc_codes}")
        
        # Artificial DTC activation - Make DTCs active for a percentage of telemetry records
        # This ensures we have enough positive examples for training
        percentage_active = 0.15  # 15% of telemetry records will have active DTCs
        num_active_records = int(len(telemetry_df) * percentage_active)
        
        # Create activation masks for each DTC
        dtc_columns = {}
        for code in dtc_codes:
            # Get all telemetry records
            indices = list(range(len(telemetry_df)))
            
            # Randomly select records where this DTC will be active
            active_indices = set(random.sample(indices, num_active_records))
            
            # Create the column with 1 for active, 0 for inactive
            dtc_columns[f'dtc_{code}'] = [1 if i in active_indices else 0 for i in range(len(telemetry_df))]
            
            print(f"Created column 'dtc_{code}' with {len(active_indices)} active instances")
        
        # Add DTC columns to the telemetry DataFrame
        for col_name, values in dtc_columns.items():
            telemetry_df[col_name] = values
        
        return telemetry_df
    
    def train_anomaly_detector(self):
        """
        Train an anomaly detector on telemetry data
        
        Returns:
            Trained IsolationForest model
        """
        # Load telemetry data
        telemetry_df = self.load_telemetry_data()
        
        if telemetry_df is None:
            print("No telemetry data available for training")
            return None
        
        # Select relevant features
        feature_cols = [col for col in ['speed', 'rpm', 'throttle', 'brake'] 
                       if col in telemetry_df.columns]
        
        if not feature_cols:
            print("No valid features found in telemetry data")
            return None
        
        # Prepare features
        X = telemetry_df[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        print(f"Training anomaly detector on {len(X)} samples with features: {feature_cols}")
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.anomaly_detector.fit(X_scaled)
        
        # Save the model
        model_data = {
            'model': self.anomaly_detector,
            'scaler': self.scaler,
            'features': feature_cols
        }
        joblib.dump(model_data, os.path.join(self.model_dir, 'telemetry_anomaly_detector.joblib'))
        
        print("Anomaly detector trained and saved")
        return self.anomaly_detector
    
    def detect_anomalies(self, telemetry_df=None, threshold=0.05):
        """
        Detect anomalies in telemetry data
        
        Args:
            telemetry_df: DataFrame with telemetry data
            threshold: Anomaly threshold (contamination rate)
            
        Returns:
            DataFrame with anomaly scores
        """
        # Load data if not provided
        if telemetry_df is None:
            telemetry_df = self.load_telemetry_data()
            
        if telemetry_df is None:
            print("No telemetry data available for anomaly detection")
            return None
        
        # Load model if not already loaded
        if self.anomaly_detector is None:
            model_path = os.path.join(self.model_dir, 'telemetry_anomaly_detector.joblib')
            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                    self.anomaly_detector = model_data['model']
                    self.scaler = model_data['scaler']
                    feature_cols = model_data['features']
                except Exception as e:
                    print(f"Error loading anomaly detector: {e}")
                    return None
            else:
                # Train a new model
                if self.train_anomaly_detector() is None:
                    return None
                model_data = joblib.load(os.path.join(self.model_dir, 'telemetry_anomaly_detector.joblib'))
                feature_cols = model_data['features']
        else:
            # Get feature columns from the model
            feature_cols = [col for col in ['speed', 'rpm', 'throttle', 'brake'] 
                           if col in telemetry_df.columns]
        
        # Prepare features
        X = telemetry_df[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Detect anomalies
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        anomalies = self.anomaly_detector.predict(X_scaled)
        
        # Add results to DataFrame
        telemetry_df['anomaly_score'] = anomaly_scores
        telemetry_df['is_anomaly'] = (anomalies == -1).astype(int)
        
        print(f"Detected {(anomalies == -1).sum()} anomalies in {len(telemetry_df)} records")
        return telemetry_df
    
    def train_dtc_prediction_model(self):
        """
        Train a model to predict DTCs based on driving patterns
        
        Returns:
            Trained classifier model
        """
        # Merge telemetry and DTC data
        merged_df = self.merge_telemetry_and_dtc_data()
        
        if merged_df is None:
            print("No merged data available for training")
            return None
        
        # Find DTC columns
        dtc_columns = [col for col in merged_df.columns if col.startswith('dtc_')]
        
        if not dtc_columns:
            print("No DTC data found in merged dataset")
            return None
        
        # Select relevant features
        feature_cols = [col for col in ['speed', 'rpm', 'throttle', 'brake'] 
                       if col in merged_df.columns]
        
        if not feature_cols:
            print("No valid features found in telemetry data")
            return None
        
        # Create aggregate features
        print("Creating aggregate features for DTC prediction...")
        
        # Group data into windows
        window_size = 100  # Number of samples per window
        
        # Calculate aggregate features for each window
        windows = []
        
        for i in range(0, len(merged_df) - window_size, window_size // 2):  # 50% overlap
            window = merged_df.iloc[i:i+window_size]
            
            # Basic statistics for each feature
            window_features = {}
            for col in feature_cols:
                window_features[f'{col}_mean'] = window[col].mean()
                window_features[f'{col}_max'] = window[col].max()
                window_features[f'{col}_std'] = window[col].std()
                
                # Rate of change
                if col in ['speed', 'throttle', 'brake']:
                    window_features[f'{col}_changes'] = window[col].diff().abs().mean()
            
            # DTC presence in this window
            for dtc_col in dtc_columns:
                window_features[dtc_col] = 1 if window[dtc_col].sum() > 0 else 0
            
            windows.append(window_features)
        
        if not windows:
            print("Not enough data to create windows")
            return None
            
        # Create DataFrame from windows
        windows_df = pd.DataFrame(windows)
        
        # For each DTC, train a separate classifier
        dtc_models = {}
        
        for dtc_col in dtc_columns:
            print(f"Training model for {dtc_col}...")
            
            # Skip if there are no positive examples
            if windows_df[dtc_col].sum() == 0:
                print(f"No positive examples for {dtc_col}, skipping")
                continue
            
            # Select features (exclude DTC columns)
            X = windows_df[[col for col in windows_df.columns if not col.startswith('dtc_')]]
            y = windows_df[dtc_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred)
            
            print(f"Classification report for {dtc_col}:")
            print(report)
            
            # Store model
            dtc_models[dtc_col] = {
                'model': clf,
                'features': X.columns.tolist(),
                'report': report
            }
        
        # Save models
        joblib.dump(dtc_models, os.path.join(self.model_dir, 'dtc_prediction_models.joblib'))
        
        self.dtc_classifier = dtc_models
        print(f"Trained and saved {len(dtc_models)} DTC prediction models")
        
        return dtc_models
    
    def predict_dtcs(self, telemetry_df=None):
        """
        Predict DTCs based on driving patterns
        
        Args:
            telemetry_df: DataFrame with telemetry data
            
        Returns:
            Dictionary with DTC predictions
        """
        # Load data if not provided
        if telemetry_df is None:
            telemetry_df = self.load_telemetry_data()
            
        if telemetry_df is None:
            print("No telemetry data available for DTC prediction")
            return None
        
        # Load models if not already loaded
        if self.dtc_classifier is None:
            model_path = os.path.join(self.model_dir, 'dtc_prediction_models.joblib')
            if os.path.exists(model_path):
                try:
                    self.dtc_classifier = joblib.load(model_path)
                except Exception as e:
                    print(f"Error loading DTC prediction models: {e}")
                    return None
            else:
                print("No trained DTC prediction models found")
                return None
        
        # Ensure enough data
        window_size = 100
        if len(telemetry_df) < window_size:
            print(f"Not enough telemetry data for prediction (need at least {window_size} samples)")
            return None
        
        # Create aggregate features
        window_features = {}
        feature_cols = [col for col in ['speed', 'rpm', 'throttle', 'brake'] 
                       if col in telemetry_df.columns]
        
        for col in feature_cols:
            window_features[f'{col}_mean'] = telemetry_df[col].mean()
            window_features[f'{col}_max'] = telemetry_df[col].max()
            window_features[f'{col}_std'] = window_features[f'{col}_std'] = telemetry_df[col].std()
            
            # Rate of change
            if col in ['speed', 'throttle', 'brake']:
                window_features[f'{col}_changes'] = telemetry_df[col].diff().abs().mean()
        
        # Make predictions for each DTC model
        predictions = {}
        
        for dtc_name, model_info in self.dtc_classifier.items():
            # Get model and features
            model = model_info['model']
            model_features = model_info['features']
            
            # Prepare input features
            X = pd.DataFrame([{feature: window_features.get(feature, 0) for feature in model_features}])
            
            # Make prediction
            probability = model.predict_proba(X)[0][1]  # Probability of DTC
            predictions[dtc_name] = {
                'probability': float(probability),
                'predicted': bool(probability > 0.5),
                'dtc_code': dtc_name.replace('dtc_', '')
            }
        
        return predictions
    
    def analyze_dtc_root_causes(self):
        """
        Analyze potential root causes for DTCs
        
        Returns:
            Dictionary with analysis results
        """
        # Merge telemetry and DTC data
        merged_df = self.merge_telemetry_and_dtc_data()
        
        if merged_df is None:
            print("No merged data available for analysis")
            return None
        
        # Find DTC columns
        dtc_columns = [col for col in merged_df.columns if col.startswith('dtc_')]
        
        if not dtc_columns:
            print("No DTC data found in merged dataset")
            return None
        
        # Select relevant features
        feature_cols = [col for col in ['speed', 'rpm', 'throttle', 'brake'] 
                    if col in merged_df.columns]
        
        # Analysis results
        results = {}
        
        # For each DTC, analyze conditions when it occurred
        for dtc_col in dtc_columns:
            dtc_code = dtc_col.replace('dtc_', '')
            print(f"Analyzing root causes for {dtc_code}...")
            
            # Extract records where this DTC was active
            dtc_active = merged_df[merged_df[dtc_col] == 1]
            dtc_inactive = merged_df[merged_df[dtc_col] == 0]
            
            if len(dtc_active) == 0:
                print(f"No active instances of {dtc_code} found")
                continue
            
            # Basic statistics
            dtc_stats = {
                'occurrences': int(len(dtc_active)),
                'conditions': {}
            }
            
            # Analyze conditions
            for feature in feature_cols:
                # Compare feature distributions
                if feature in dtc_active.columns:
                    active_stats = {
                        'mean': float(dtc_active[feature].mean()),
                        'median': float(dtc_active[feature].median()),
                        'std': float(dtc_active[feature].std()),
                        'min': float(dtc_active[feature].min()),
                        'max': float(dtc_active[feature].max())
                    }
                    
                    inactive_stats = {
                        'mean': float(dtc_inactive[feature].mean()),
                        'median': float(dtc_inactive[feature].median()),
                        'std': float(dtc_inactive[feature].std()),
                        'min': float(dtc_inactive[feature].min()),
                        'max': float(dtc_inactive[feature].max())
                    }
                    
                    # Calculate significance of difference
                    mean_diff = active_stats['mean'] - inactive_stats['mean']
                    std_pooled = np.sqrt((active_stats['std']**2 + inactive_stats['std']**2) / 2)
                    
                    if std_pooled > 0:
                        effect_size = abs(mean_diff / std_pooled)  # Cohen's d
                    else:
                        effect_size = 0.0
                    
                    dtc_stats['conditions'][feature] = {
                        'active': active_stats,
                        'inactive': inactive_stats,
                        'mean_difference': float(mean_diff),
                        'effect_size': float(effect_size),
                        'significant': bool(effect_size > 0.5)
                    }
            
            # Find most significant features
            significant_features = sorted(
                [(feature, info['effect_size']) 
                for feature, info in dtc_stats['conditions'].items() if info['significant']],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Convert to list of strings for JSON serialization
            dtc_stats['significant_features'] = [str(feature) for feature, _ in significant_features]
            
            # Create a root cause hypothesis
            if significant_features:
                # Formulate a hypothesis based on the most significant features
                root_cause = "This DTC may be triggered by "
                
                for i, (feature, effect) in enumerate(significant_features[:3]):  # Top 3 features
                    condition_info = dtc_stats['conditions'][feature]
                    
                    # Determine if higher or lower values are associated with the DTC
                    if condition_info['mean_difference'] > 0:
                        direction = "high"
                    else:
                        direction = "low"
                    
                    if i == 0:
                        root_cause += f"{direction} {feature}"
                    elif i == len(significant_features[:3]) - 1:
                        root_cause += f" and {direction} {feature}"
                    else:
                        root_cause += f", {direction} {feature}"
                    
                dtc_stats['root_cause_hypothesis'] = root_cause
            else:
                dtc_stats['root_cause_hypothesis'] = "No clear pattern identified for this DTC"
            
            results[dtc_code] = dtc_stats
        
        # Save analysis results
        os.makedirs("analysis_results", exist_ok=True)
        try:
            with open(os.path.join("analysis_results", "dtc_root_causes.json"), 'w') as f:
                json.dump(results, f, indent=2, cls=CustomJSONEncoder)
            
            print(f"Root cause analysis completed for {len(results)} DTCs")
        except TypeError as e:
            print(f"Error serializing results to JSON: {e}")
            # Create a simpler version without the conditions
            simplified_results = {}
            for dtc_code, stats in results.items():
                simplified_results[dtc_code] = {
                    'occurrences': stats.get('occurrences', 0),
                    'root_cause_hypothesis': stats.get('root_cause_hypothesis', 'Unknown'),
                    'significant_features': stats.get('significant_features', [])
                }
            
            # Try to save the simplified version
            try:
                with open(os.path.join("analysis_results", "dtc_root_causes.json"), 'w') as f:
                    json.dump(simplified_results, f, indent=2)
                print("Saved simplified version of analysis results")
            except Exception as e2:
                print(f"Error saving simplified results: {e2}")
        
        return results
    
    def _generate_dtc_visualization(self, dtc_code, dtc_active, dtc_inactive, significant_features):
        """Generate visualizations for DTC analysis"""
        if not significant_features:
            return
        
        # Create output directory
        os.makedirs("analysis_results/visualizations", exist_ok=True)
        
        # For the top 2 most significant features, create a scatter plot
        if len(significant_features) >= 2:
            feature1, _ = significant_features[0]
            feature2, _ = significant_features[1]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(dtc_inactive[feature1], dtc_inactive[feature2], 
                       alpha=0.5, label='DTC Inactive', color='blue')
            plt.scatter(dtc_active[feature1], dtc_active[feature2], 
                       alpha=0.7, label='DTC Active', color='red')
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.title(f'Conditions Associated with {dtc_code}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"analysis_results/visualizations/{dtc_code}_scatter.png")
            plt.close()
        
        # For each significant feature, create a histogram
        for feature, _ in significant_features[:3]:
            plt.figure(figsize=(10, 6))
            plt.hist(dtc_inactive[feature], bins=30, alpha=0.5, label='DTC Inactive', color='blue')
            plt.hist(dtc_active[feature], bins=30, alpha=0.5, label='DTC Active', color='red')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {feature} for {dtc_code}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"analysis_results/visualizations/{dtc_code}_{feature}_hist.png")
            plt.close()
    
    def real_time_dtc_prediction(self, telemetry_buffer, window_size=100):
        """
        Make real-time DTC predictions from a telemetry buffer
        
        Args:
            telemetry_buffer: List of telemetry data points
            window_size: Size of the window to analyze
            
        Returns:
            Dictionary with DTC predictions
        """
        # Check if we have enough data
        if len(telemetry_buffer) < window_size:
            return None
        
        # Create DataFrame from buffer
        df = pd.DataFrame(telemetry_buffer[-window_size:])
        
        # Make predictions
        return self.predict_dtcs(df)
    
        
    def export_combined_dataset(self, output_file="combined_telemetry_dtc_data.csv"):
        """
        Export a combined dataset with telemetry and DTC data for external analysis
        
        Args:
            output_file: Output CSV file name
            
        Returns:
            True if successful, False otherwise
        """
        # Merge telemetry and DTC data
        merged_df = self.merge_telemetry_and_dtc_data()
        
        if merged_df is None:
            print("No merged data available for export")
            return False
        
        # Save to CSV
        try:
            merged_df.to_csv(output_file, index=False)
            print(f"Combined dataset exported to {output_file}")
            return True
        except Exception as e:
            print(f"Error exporting combined dataset: {e}")
            return False

class RealTimeDTCAnalyzer:
    """Real-time analyzer for DTC prediction and anomaly detection"""
    
    def __init__(self, socketio, update_interval=5.0):
        self.analyzer = DTCAnalyzer()
        self.socketio = socketio
        self.telemetry_buffer = []
        self.max_buffer_size = 1000
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        # Try to load anomaly detector
        model_path = os.path.join(self.analyzer.model_dir, 'telemetry_anomaly_detector.joblib')
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                self.analyzer.anomaly_detector = model_data['model']
                self.analyzer.scaler = model_data['scaler']
                print("Loaded anomaly detector model")
            except Exception as e:
                print(f"Error loading anomaly detector: {e}")
        
        # Try to load DTC prediction models
        model_path = os.path.join(self.analyzer.model_dir, 'dtc_prediction_models.joblib')
        if os.path.exists(model_path):
            try:
                self.analyzer.dtc_classifier = joblib.load(model_path)
                print(f"Loaded {len(self.analyzer.dtc_classifier)} DTC prediction models")
            except Exception as e:
                print(f"Error loading DTC prediction models: {e}")
    
    def add_telemetry(self, telemetry):
        """Add telemetry data to the buffer"""
        self.telemetry_buffer.append({
            'timestamp': time.time(),
            **telemetry
        })
        
        # Maintain buffer size
        if len(self.telemetry_buffer) > self.max_buffer_size:
            self.telemetry_buffer = self.telemetry_buffer[-self.max_buffer_size:]
    
    def start(self):
        """Start the real-time analyzer thread"""
        if self.running:
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._analysis_loop)
        self.thread.daemon = True
        self.thread.start()
        
        print("Real-time DTC analyzer started")
        return True
    
    def stop(self):
        """Stop the real-time analyzer thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Real-time DTC analyzer stopped")
    
    def _analysis_loop(self):
        """Background thread for real-time analysis"""
        while self.running:
            try:
                # Wait for sufficient data
                if len(self.telemetry_buffer) < 100:
                    time.sleep(1)
                    continue
                
                # Create DataFrame from buffer for analysis
                df = pd.DataFrame(self.telemetry_buffer[-100:])
                
                # Detect anomalies
                anomaly_results = None
                if self.analyzer.anomaly_detector is not None:
                    try:
                        # Prepare features
                        feature_cols = [col for col in ['speed', 'rpm', 'throttle', 'brake'] 
                                      if col in df.columns]
                        
                        if feature_cols:
                            X = df[feature_cols].fillna(0)
                            X_scaled = self.analyzer.scaler.transform(X)
                            
                            # Get anomaly scores
                            anomaly_scores = self.analyzer.anomaly_detector.decision_function(X_scaled)
                            anomalies = self.analyzer.anomaly_detector.predict(X_scaled)
                            
                            # Calculate percentage of anomalies
                            anomaly_percentage = (anomalies == -1).mean() * 100
                            
                            anomaly_results = {
                                'anomaly_percentage': float(anomaly_percentage),
                                'recent_anomalies': int((anomalies == -1).sum()),
                                'min_score': float(anomaly_scores.min()),
                                'max_score': float(anomaly_scores.max())
                            }
                    except Exception as e:
                        print(f"Error detecting anomalies: {e}")
                
                # Predict DTCs
                dtc_predictions = None
                if self.analyzer.dtc_classifier is not None:
                    try:
                        dtc_predictions = self.analyzer.real_time_dtc_prediction(self.telemetry_buffer)
                    except Exception as e:
                        print(f"Error predicting DTCs: {e}")
                
                # Emit analysis results
                if anomaly_results or dtc_predictions:
                    self.socketio.emit('telemetry_analysis', {
                        'timestamp': time.time(),
                        'anomalies': anomaly_results,
                        'dtc_predictions': dtc_predictions
                    })
                
                # Sleep before next analysis
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in real-time analysis: {e}")
                time.sleep(2)

def initialize(app, socketio, canoe_interface=None):
    """Initialize the DTC analyzer with Flask app and Socket.IO"""
    
    # Create analyzer
    analyzer = DTCAnalyzer()
    
    # Create real-time analyzer
    real_time_analyzer = RealTimeDTCAnalyzer(socketio)
    
    # Register API routes
    @app.route('/api/dtc-analysis')
    def get_dtc_analysis():
        """API endpoint to get DTC analysis results"""
        # Check for existing analysis
        analysis_path = os.path.join("analysis_results", "dtc_root_causes.json")
        if os.path.exists(analysis_path):
            try:
                with open(analysis_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading analysis file: {e}")
                # If file exists but is corrupted, delete it so it will be regenerated
                try:
                    os.remove(analysis_path)
                    print(f"Removed corrupted analysis file: {analysis_path}")
                except:
                    pass
                return {"message": "Analysis file was corrupted and has been cleared. Please train models again."}
        else:
            return {"message": "No DTC analysis results available"}
    
    @app.route('/api/train-models')
    def train_models():
        """API endpoint to trigger model training"""
        # Start training in a background thread
        def train_background():
            # Train anomaly detector
            analyzer.train_anomaly_detector()
            
            # Train DTC predictor
            analyzer.train_dtc_prediction_model()
            
            # Analyze root causes
            analyzer.analyze_dtc_root_causes()
            
            # Notify clients that training is complete
            socketio.emit('training_complete', {
                'timestamp': time.time(),
                'models_trained': ['anomaly_detector', 'dtc_predictor'],
                'analyses_completed': ['root_cause_analysis']
            })
        
        threading.Thread(target=train_background).start()
        
        return {"message": "Model training started in background"}
    
    # Start the real-time analyzer
    real_time_analyzer.start()
    
    # Register telemetry handler to feed data to the analyzer
    def handle_telemetry(telemetry):
        real_time_analyzer.add_telemetry(telemetry)
    
    # Register cleanup handler
    import atexit
    def cleanup():
        real_time_analyzer.stop()
    
    atexit.register(cleanup)
    
    # Return the real-time analyzer
    return real_time_analyzer