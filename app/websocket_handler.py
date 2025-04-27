"""
WebSocket handler for Forza Horizon 5 Telemetry Dashboard

This module handles WebSocket connections for the dashboard.
"""
import os
from .data_logger import export_telemetry, clear_telemetry

def register_handlers(socketio):
    """Register event handlers for Socket.IO"""
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
        
    @socketio.on('export_telemetry')
    def handle_export():
        """Handle a manual telemetry export request"""
        print('Manual telemetry export requested')
        export_telemetry()
        
        # Find the most recent export file
        export_dir = 'telemetry_exports'
        if os.path.exists(export_dir):
            files = [os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith('.csv')]
            if files:
                latest_file = max(files, key=os.path.getmtime)
                filename = os.path.basename(latest_file)
                socketio.emit('export_complete', {'filename': filename})
                return
                
        socketio.emit('export_complete', {'filename': 'Unknown'})
        
    @socketio.on('clear_telemetry')
    def handle_clear():
        """Handle a clear telemetry data request"""
        print('Clearing telemetry data')
        clear_telemetry()
        socketio.emit('clear_complete')