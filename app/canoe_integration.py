"""
CANoe Integration Module for Forza Dashboard

This module enables integration with Vector CANoe to collect and analyze
diagnostic trouble codes (DTCs) alongside Forza telemetry data.
"""

import threading
import time
import json
import os
from datetime import datetime
import pythoncom
from win32com.client import Dispatch, constants

class CANoeInterface:
    """Interface for communicating with Vector CANoe"""
    
    def __init__(self):
        self.app = None
        self.measurement = None
        self.active_dtcs = {}
        self.connected = False
        self.dtc_lock = threading.Lock()
        self.canoe_path = None  # Path to CANoe configuration
        
    def connect(self, canoe_path=None):
        """Connect to CANoe application"""
        try:
            # Initialize COM for this thread
            pythoncom.CoInitialize()
            
            # Store configuration path
            self.canoe_path = canoe_path
            
            # Get CANoe application
            self.app = Dispatch("CANoe.Application")
            print(f"Connected to CANoe version: {self.app.Version}")
            
            # Get measurement interface
            self.measurement = self.app.Measurement
            
            # Load configuration if provided
            if canoe_path and os.path.exists(canoe_path):
                print(f"Loading CANoe configuration: {canoe_path}")
                self.app.Open(canoe_path)
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Error connecting to CANoe: {e}")
            self.connected = False
            return False
    
    def start_measurement(self):
        """Start CANoe measurement"""
        try:
            if not self.connected or not self.measurement:
                print("CANoe not connected")
                return False
                
            if self.measurement.Running:
                print("Measurement already running")
                return True
                
            self.measurement.Start()
            print("CANoe measurement started")
            return True
            
        except Exception as e:
            print(f"Error starting CANoe measurement: {e}")
            return False
    
    def stop_measurement(self):
        """Stop CANoe measurement"""
        try:
            if not self.connected or not self.measurement:
                return False
                
            if not self.measurement.Running:
                return True
                
            self.measurement.Stop()
            print("CANoe measurement stopped")
            return True
            
        except Exception as e:
            print(f"Error stopping CANoe measurement: {e}")
            return False
    
    def get_dtc_data(self):
        """Get DTC data from CANoe"""
        try:
            if not self.connected:
                return None
            
            # Test Data for DTCs (for development purposes)
            # In a real implementation, this would come from CANoe
            sample_dtcs = {
                "P0300": {
                    "description": "Random/Multiple Cylinder Misfire Detected",
                    "timestamp": time.time(),
                    "status": "active"
                },
                "P0171": {
                    "description": "Fuel System Too Lean (Bank 1)",
                    "timestamp": time.time() - 60,
                    "status": "pending"
                }
            }
            
            return sample_dtcs
            
        except Exception as e:
            print(f"Error getting DTC data from CANoe: {e}")
            return None
    
    def update_dtcs(self):
        """Update the active DTCs from CANoe"""
        with self.dtc_lock:
            dtcs = self.get_dtc_data()
            if dtcs:
                self.active_dtcs = dtcs
    
    def get_active_dtcs(self):
        """Get the active DTCs"""
        with self.dtc_lock:
            return self.active_dtcs
    
    def add_dtc(self, code, description, status="active"):
        """Manually add a DTC (for testing)"""
        with self.dtc_lock:
            self.active_dtcs[code] = {
                "description": description,
                "timestamp": time.time(),
                "status": status
            }
    
    def clear_dtcs(self):
        """Clear all DTCs"""
        with self.dtc_lock:
            self.active_dtcs = {}
    
    def disconnect(self):
        """Disconnect from CANoe"""
        try:
            if self.connected:
                if self.measurement and self.measurement.Running:
                    self.measurement.Stop()
                # Close CANoe
                self.app.Quit()
                self.app = None
                self.measurement = None
                self.connected = False
                print("Disconnected from CANoe")
                
                # Uninitialize COM
                pythoncom.CoUninitialize()
            
            return True
            
        except Exception as e:
            print(f"Error disconnecting from CANoe: {e}")
            return False

class CANoeDTCMonitor:
    """Background monitor for DTCs from CANoe"""
    
    def __init__(self, interface, socketio):
        self.interface = interface
        self.socketio = socketio
        self.running = False
        self.thread = None
        self.monitor_interval = 1.0  # Check interval in seconds
        
    def start(self):
        """Start the DTC monitor thread"""
        if self.running:
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
        print("CANoe DTC monitor started")
        return True
        
    def stop(self):
        """Stop the DTC monitor thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("CANoe DTC monitor stopped")
        
    def _monitor_loop(self):
        """Background thread for monitoring DTCs"""
        last_dtcs = {}
        
        while self.running:
            try:
                # Update DTCs from CANoe
                self.interface.update_dtcs()
                
                # Get current DTCs
                current_dtcs = self.interface.get_active_dtcs()
                
                # Check for changes
                if current_dtcs != last_dtcs:
                    # Find new DTCs
                    new_dtcs = {code: info for code, info in current_dtcs.items() 
                               if code not in last_dtcs}
                    
                    # Find cleared DTCs
                    cleared_dtcs = {code: info for code, info in last_dtcs.items() 
                                  if code not in current_dtcs}
                    
                    # Emit events if there are changes
                    if new_dtcs:
                        self.socketio.emit('dtc_new', {
                            'dtcs': new_dtcs,
                            'count': len(new_dtcs)
                        })
                        print(f"New DTCs detected: {list(new_dtcs.keys())}")
                    
                    if cleared_dtcs:
                        self.socketio.emit('dtc_cleared', {
                            'dtcs': cleared_dtcs,
                            'count': len(cleared_dtcs)
                        })
                        print(f"DTCs cleared: {list(cleared_dtcs.keys())}")
                    
                    # Always emit the current state
                    self.socketio.emit('dtc_update', {
                        'dtcs': current_dtcs,
                        'count': len(current_dtcs),
                        'timestamp': time.time()
                    })
                    
                    # Update the last state
                    last_dtcs = current_dtcs.copy()
                
            except Exception as e:
                print(f"Error in DTC monitor: {e}")
            
            # Sleep before next check
            time.sleep(self.monitor_interval)

class DTCLogger:
    """Logger for DTC events"""
    
    def __init__(self, log_dir="dtc_logs"):
        self.log_dir = log_dir
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
    def log_dtc_event(self, event_type, dtcs):
        """Log a DTC event to CSV and JSON"""
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create event record
            event = {
                "timestamp": timestamp,
                "event_type": event_type,
                "dtcs": dtcs
            }
            
            # Log to JSON
            json_path = os.path.join(self.log_dir, f"dtc_event_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump(event, f, indent=2)
            
            # Log to central CSV
            csv_path = os.path.join(self.log_dir, "dtc_events.csv")
            csv_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a') as f:
                # Write header if new file
                if not csv_exists:
                    f.write("timestamp,event_type,dtc_code,description,status\n")
                
                # Write each DTC on a separate line
                for code, info in dtcs.items():
                    f.write(f"{timestamp},{event_type},{code},\"{info.get('description', '')}\",{info.get('status', '')}\n")
            
            return True
            
        except Exception as e:
            print(f"Error logging DTC event: {e}")
            return False

def initialize(app, socketio):
    """Initialize the CANoe integration with Flask app and Socket.IO"""
    
    # Create CANoe interface
    interface = CANoeInterface()
    
    # Create DTC monitor
    monitor = CANoeDTCMonitor(interface, socketio)
    
    # Create DTC logger
    logger = DTCLogger()
    
    # Try to connect to CANoe
    canoe_path = os.environ.get('CANOE_CONFIG_PATH')  # Get from environment variable
    if interface.connect(canoe_path):
        # Start measurement
        interface.start_measurement()
        
        # Start DTC monitor
        monitor.start()
    else:
        print("Running without CANoe connection")
        # Add some test DTCs for development
        interface.add_dtc("P0300", "Random/Multiple Cylinder Misfire Detected")
        interface.add_dtc("P0171", "Fuel System Too Lean (Bank 1)", "pending")
    
    # Register Socket.IO handlers
    @socketio.on('clear_dtcs')
    def handle_clear_dtcs():
        """Handle a request to clear DTCs"""
        print("Clearing DTCs")
        dtcs = interface.get_active_dtcs()
        if dtcs:
            # Log the clear event
            logger.log_dtc_event("cleared", dtcs)
            
            # Clear DTCs
            interface.clear_dtcs()
            
            return {'status': 'ok', 'message': 'DTCs cleared'}
        else:
            return {'status': 'info', 'message': 'No DTCs to clear'}
    
    @socketio.on('add_test_dtc')
    def handle_add_test_dtc(data):
        """Handle a request to add a test DTC (for development)"""
        code = data.get('code')
        description = data.get('description', 'Test DTC')
        status = data.get('status', 'active')
        
        if code:
            interface.add_dtc(code, description, status)
            
            # Log the event
            logger.log_dtc_event("added", {code: {
                'description': description,
                'status': status,
                'timestamp': time.time()
            }})
            
            return {'status': 'ok', 'message': f'Added test DTC: {code}'}
        else:
            return {'status': 'error', 'message': 'No DTC code provided'}
    
    # Register API routes
    @app.route('/api/dtcs')
    def get_dtcs():
        """API endpoint to get active DTCs"""
        return interface.get_active_dtcs()
    
    # Register cleanup handler
    import atexit
    def cleanup():
        """Clean up resources when the application shuts down"""
        print("Cleaning up CANoe resources...")
        monitor.stop()
        interface.stop_measurement()
        interface.disconnect()
    
    atexit.register(cleanup)
    
    # Return the interface for use in other parts of the application
    return interface