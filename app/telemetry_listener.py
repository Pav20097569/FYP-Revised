"""
Forza Horizon 5 Telemetry Listener

This module listens for UDP telemetry data from Forza Horizon 5 and logs it.
"""

import socket
import time
import threading
from .data_parser import parse_packet
from .data_logger import log_telemetry

def listen_for_telemetry(socketio):
    """Listen for UDP telemetry data from Forza Horizon 5"""
    # Forza sends data on UDP
    UDP_IP = "127.0.0.1"  # Listen only on localhost
    UDP_PORT = 5300  # Default Forza data port

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print(f"Listening for Forza telemetry on {UDP_IP}:{UDP_PORT}")
    print("Telemetry will be logged and exported to CSV when the application closes")

    # Update rate control (emits every 100ms for smoother UI updates)
    UPDATE_INTERVAL = 0.1  # seconds
    last_emit_time = time.time()
    
    # Store the last valid data to handle occasional bad packets
    last_valid_data = None
    
    # Track packet size for debugging
    packet_sizes = set()
    last_size_report_time = time.time()

    while True:
        try:
            # Receive packet with a timeout to prevent blocking forever
            sock.settimeout(1.0)
            data, addr = sock.recvfrom(2048)  # Use larger buffer for all packet sizes
            current_time = time.time()
            
            # Keep track of different packet sizes we're seeing
            packet_sizes.add(len(data))
            
            # Report packet sizes every 20 seconds for debugging
            if current_time - last_size_report_time > 20:
                print(f"Observed packet sizes: {sorted(packet_sizes)}")
                last_size_report_time = current_time
            
            # Only process and emit if enough time has passed
            if current_time - last_emit_time >= UPDATE_INTERVAL:
                parsed = parse_packet(data)
                
                if parsed:
                    # Store this as our last valid data
                    last_valid_data = parsed
                    
                    # Log the telemetry data for CSV export
                    log_telemetry(parsed)
                    
                    # Emit to clients
                    socketio.emit('telemetry', parsed)
                    last_emit_time = current_time
                elif last_valid_data:
                    # If we have last valid data but current parse failed, 
                    # continue sending the last valid data (prevents flickering)
                    socketio.emit('telemetry', last_valid_data)
                    last_emit_time = current_time
                
        except socket.timeout:
            # This is normal, just continue
            pass
        except Exception as e:
            print(f"Error in telemetry listener: {e}")
            # Short sleep to prevent CPU spinning on continuous errors
            time.sleep(0.1)

def start_listener(socketio):
    """Start the telemetry listener in a background thread"""
    thread = threading.Thread(target=listen_for_telemetry, args=(socketio,))
    thread.daemon = True
    thread.start()
    print("Telemetry listener started in background thread")