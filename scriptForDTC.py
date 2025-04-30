"""
DTC Generator Script for Testing

This script creates multiple test DTCs with proper timestamps that align with telemetry data.
It will explicitly set timestamps to numeric Unix time values to solve timestamp parsing issues.
"""

import os
import time
import json
import csv
import random
from datetime import datetime, timedelta

# List of common DTC codes and descriptions
SAMPLE_DTCS = [
    {"code": "P0300", "description": "Random/Multiple Cylinder Misfire Detected", "status": "active"},
    {"code": "P0171", "description": "Fuel System Too Lean (Bank 1)", "status": "pending"},
    {"code": "P0420", "description": "Catalyst System Efficiency Below Threshold", "status": "active"},
    {"code": "P0455", "description": "Evaporative Emission System Leak Detected", "status": "active"},
    {"code": "P0102", "description": "Mass Air Flow Sensor Circuit Low Input", "status": "pending"},
    {"code": "P0131", "description": "O2 Sensor Circuit Low Voltage", "status": "active"},
    {"code": "P0302", "description": "Cylinder 2 Misfire Detected", "status": "active"},
    {"code": "P0401", "description": "Exhaust Gas Recirculation Flow Insufficient", "status": "pending"}
]

def create_test_dtcs():
    """Create multiple test DTCs that align with telemetry data"""
    # Ensure directories exist
    os.makedirs('dtc_logs', exist_ok=True)
    
    # Find telemetry file(s)
    telemetry_dir = 'telemetry_exports'
    if not os.path.exists(telemetry_dir):
        print(f"Error: Telemetry directory '{telemetry_dir}' not found")
        return
    
    telemetry_files = [f for f in os.listdir(telemetry_dir) if f.endswith('.csv')]
    if not telemetry_files:
        print(f"Error: No telemetry CSV files found in '{telemetry_dir}'")
        return
    
    # Use the most recent telemetry file
    telemetry_file = sorted(telemetry_files)[-1]
    telemetry_path = os.path.join(telemetry_dir, telemetry_file)
    
    # Get telemetry timestamp range
    telemetry_timestamps = []
    with open(telemetry_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'timestamp' in row:
                try:
                    telemetry_timestamps.append(float(row['timestamp']))
                except ValueError:
                    pass
    
    if not telemetry_timestamps:
        print(f"Error: No valid timestamps found in telemetry file '{telemetry_file}'")
        return
    
    # Use timestamps from the telemetry data
    telemetry_timestamps.sort()
    telemetry_start = telemetry_timestamps[0]
    telemetry_end = telemetry_timestamps[-1]
    telemetry_range = telemetry_end - telemetry_start
    
    # Create DTCs spread across the telemetry time range
    dtcs_to_create = random.sample(SAMPLE_DTCS, min(5, len(SAMPLE_DTCS)))
    
    # Create a CSV file
    csv_path = os.path.join('dtc_logs', 'dtc_events.csv')
    
    # Delete any existing DTC CSV to start fresh
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Deleted existing {csv_path}")
    
    # Open new CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'formatted_time', 'event_type', 'dtc_code', 'description', 'status'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Create DTC records
        for i, dtc in enumerate(dtcs_to_create):
            # Calculate a position in the telemetry range (20-80% to avoid edge cases)
            position = 0.2 + (0.6 * (i / max(1, len(dtcs_to_create) - 1)))
            dtc_timestamp = telemetry_start + (telemetry_range * position)
            
            # Format time for readability
            formatted_time = datetime.fromtimestamp(dtc_timestamp).strftime("%Y%m%d_%H%M%S")
            
            # Write to CSV
            writer.writerow({
                'timestamp': dtc_timestamp,  # This is a numeric Unix timestamp
                'formatted_time': formatted_time,
                'event_type': 'added',
                'dtc_code': dtc['code'],
                'description': dtc['description'],
                'status': dtc['status']
            })
            
            # Create a JSON file for this DTC
            event = {
                "timestamp": dtc_timestamp,  # This is a numeric Unix timestamp
                "formatted_time": formatted_time,
                "event_type": "added",
                "dtcs": {
                    dtc['code']: {
                        'description': dtc['description'],
                        'status': dtc['status'],
                        'timestamp': dtc_timestamp
                    }
                }
            }
            
            json_path = os.path.join('dtc_logs', f"dtc_event_{formatted_time}.json")
            with open(json_path, 'w') as f:
                json.dump(event, f, indent=2)
            
            print(f"Created DTC {dtc['code']} at timestamp {dtc_timestamp} ({formatted_time})")
    
    print(f"\nCreated {len(dtcs_to_create)} DTCs spread across the telemetry timeline")
    print(f"CSV file: {csv_path}")
    print("\nYou can now run the model training to see if it picks up these DTCs")

if __name__ == "__main__":
    create_test_dtcs()