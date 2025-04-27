"""
Forza Horizon 5 Telemetry Data Parser

This module parses UDP telemetry data from Forza Horizon 5.
"""

import struct
import math
from .forza_packet_format import FH5_PACKET_FORMAT, FH5_PACKET_STRUCT, get_gear_display

class ForzaDataParser:
    def __init__(self):
        # Use the format definition from the imported module
        self.format_definition = FH5_PACKET_FORMAT
        self.packet_struct = FH5_PACKET_STRUCT
        
    def parse_packet(self, data):
        """Parse a UDP packet from Forza Horizon 5"""
        try:
            # Check packet size
            if len(data) != self.packet_struct['total_size']:
                print(f"Unexpected packet size: {len(data)} bytes (expected {self.packet_struct['total_size']})")
                return None
                
            # Parse the data
            primitive_size = self.packet_struct['primitive_count'] * 4  # Each float is 4 bytes
            
            # Parse primitive values (floats)
            primitive_values = struct.unpack(
                self.packet_struct['primitive_format'], 
                data[:primitive_size]
            )
            
            # Parse remaining bytes (non-float values)
            remaining_format = self.packet_struct['remaining_format']
            remaining_size = struct.calcsize(remaining_format)
            remaining_values = struct.unpack(
                remaining_format, 
                data[primitive_size:primitive_size + remaining_size]
            )
            
            # Combine all values
            all_values = primitive_values + remaining_values
            
            # Check if game is active
            if all_values[self.format_definition['IsRaceOn']['index']] == 0:
                return None
                
            # Extract values with fallbacks
            try:
                rpm = all_values[self.format_definition['CurrentEngineRpm']['index']]
            except (IndexError, KeyError):
                rpm = 0
                
            try:
                gear_value = all_values[self.format_definition['Gear']['index']]
                gear = get_gear_display(gear_value)
            except (IndexError, KeyError):
                gear = "N"  # Default to neutral
                
            try:
                throttle = all_values[self.format_definition['Throttle']['index']]
                throttle = round(throttle / 255, 2) if self.format_definition['Throttle'].get('normalize') else throttle
            except (IndexError, KeyError):
                throttle = 0
                
            try:
                brake = all_values[self.format_definition['Brake']['index']]
                brake = round(brake / 255, 2) if self.format_definition['Brake'].get('normalize') else brake
            except (IndexError, KeyError):
                brake = 0
                
            # Calculate speed from velocity components
            try:
                vx = all_values[self.format_definition['VelocityX']['index']]
                vy = all_values[self.format_definition['VelocityY']['index']]
                vz = all_values[self.format_definition['VelocityZ']['index']]
                
                # Calculate speed magnitude
                speed_ms = math.sqrt(vx*vx + vy*vy + vz*vz)
                
                # Use direct speed value if it seems reasonable
                direct_speed = all_values[self.format_definition['Speed']['index']]
                if 0 <= direct_speed < 150:  # 150 m/s = ~540 km/h (reasonable max)
                    speed_ms = direct_speed
            except (IndexError, KeyError):
                speed_ms = 0
                
            # Return the parsed data
            return {
                'speed': round(speed_ms * 3.6, 1),  # Convert m/s to km/h
                'rpm': round(rpm),
                'gear': gear,
                'throttle': throttle,
                'brake': brake
            }
            
        except Exception as e:
            print(f"Error parsing Forza data: {e}")
            return None

# Create a singleton instance
parser = ForzaDataParser()

def parse_packet(data):
    """Helper function that uses the singleton parser"""
    return parser.parse_packet(data)