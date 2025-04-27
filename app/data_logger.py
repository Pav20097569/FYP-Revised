"""
Forza Telemetry Data Logger

This module logs telemetry data and exports it to CSV when the application closes.
"""

import csv
import os
import time
import atexit
import threading
from datetime import datetime

class TelemetryLogger:
    def __init__(self, max_records=10000):
        """Initialize the telemetry logger
        
        Args:
            max_records: Maximum number of telemetry records to keep in memory
        """
        self.telemetry_data = []
        self.max_records = max_records
        self.log_lock = threading.Lock()
        self.export_dir = "telemetry_exports"
        
        # Ensure export directory exists
        os.makedirs(self.export_dir, exist_ok=True)
        
        # Register the export function to run when the program exits
        atexit.register(self.export_to_csv)
        
        print(f"Telemetry logger initialized (max records: {max_records})")
    
    def log_telemetry(self, telemetry):
        """Log a telemetry data point
        
        Args:
            telemetry: Dictionary containing telemetry data
        """
        with self.log_lock:
            # Add timestamp to telemetry data
            timestamped_data = {
                'timestamp': time.time(),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                **telemetry
            }
            
            # Add to the telemetry data list
            self.telemetry_data.append(timestamped_data)
            
            # If we've exceeded the maximum number of records, remove the oldest one
            if len(self.telemetry_data) > self.max_records:
                self.telemetry_data.pop(0)
    
    def export_to_csv(self):
        """Export the logged telemetry data to a CSV file"""
        if not self.telemetry_data:
            print("No telemetry data to export")
            return
            
        try:
            # Generate a filename with the current timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.export_dir, f"forza_telemetry_{timestamp}.csv")
            
            print(f"Exporting {len(self.telemetry_data)} telemetry records to {filename}")
            
            # Get the field names from the first record
            with self.log_lock:
                fieldnames = list(self.telemetry_data[0].keys())
                
                # Write the data to CSV
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.telemetry_data)
            
            print(f"Telemetry data exported successfully to {filename}")
        except Exception as e:
            print(f"Error exporting telemetry data: {e}")
    
    def manual_export(self):
        """Manually trigger an export of the logged telemetry data"""
        self.export_to_csv()
        
    def clear_data(self):
        """Clear all logged telemetry data"""
        with self.log_lock:
            self.telemetry_data = []
        print("Telemetry data cleared")

# Create a singleton instance
logger = TelemetryLogger()

def log_telemetry(telemetry):
    """Helper function that uses the singleton logger"""
    logger.log_telemetry(telemetry)

def export_telemetry():
    """Helper function to manually export telemetry"""
    logger.export_to_csv()

def clear_telemetry():
    """Helper function to clear telemetry data"""
    logger.clear_data()