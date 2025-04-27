"""
Enhanced Forza Horizon 5 Telemetry Data Parser with Strict Validation

This module provides a more aggressive parser for Forza telemetry data,
with multiple validation layers and strong smoothing to eliminate speed spikes.
"""

import struct
import math
import time
from collections import deque
from .forza_packet_format import FH5_PACKET_FORMAT, FH5_PACKET_STRUCT, get_gear_display

class ForzaDataParser:
    def __init__(self):
        # Use the format definition from the imported module
        self.format_definition = FH5_PACKET_FORMAT
        self.packet_struct = FH5_PACKET_STRUCT
        
        # Speed validation and smoothing
        self.speed_history = deque(maxlen=10)  # Increased buffer size
        self.speed_median_history = deque(maxlen=5)  # For median filtering
        self.last_valid_speed = 0.0
        self.last_update_time = time.time()
        
        # Strict limits
        self.MAX_SPEED_CHANGE = 20.0  # Much stricter limit (km/h per update)
        self.MAX_REASONABLE_SPEED = 500.0  # km/h
        self.MIN_REASONABLE_SPEED = 0.0  # km/h
        
        # Smoothing weights - higher weight to history means more smoothing
        self.HISTORY_WEIGHT = 0.85  # 85% history, 15% new value
        
        # Additional state for better filtering
        self.consecutive_spikes = 0  # Count consecutive spike detections
        self.last_stable_speed = 0.0  # Last speed that wasn't flagged as a spike
        self.in_steady_state = False  # Flag for when car is moving at relatively constant speed
        
        # RPM validation
        self.rpm_history = deque(maxlen=5)
        self.last_valid_rpm = 0.0
        
        # Packet validation
        self.valid_packet_count = 0
        self.invalid_packet_count = 0
        
    def _get_median(self, values):
        """Calculate median of a list of values"""
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n == 0:
            return 0
        if n % 2 == 1:
            return sorted_values[n // 2]
        else:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    
    def _validate_speed(self, raw_speed):
        """Multi-stage speed validation and smoothing to eliminate spikes"""
        # Convert to km/h
        speed_kmh = raw_speed * 3.6
        
        # Stage 1: Basic range check
        if speed_kmh < self.MIN_REASONABLE_SPEED or speed_kmh > self.MAX_REASONABLE_SPEED:
            self.consecutive_spikes += 1
            print(f"Speed out of range: {speed_kmh:.1f} km/h")
            return self._get_fallback_speed()
        
        # Stage 2: Acceleration/deceleration rate check
        if self.speed_history:
            last_speed = self.speed_history[-1]
            speed_change = abs(speed_kmh - last_speed)
            
            current_time = time.time()
            time_elapsed = max(0.001, current_time - self.last_update_time)  # Avoid division by zero
            self.last_update_time = current_time
            
            # Calculate maximum allowed change based on time elapsed
            max_change = self.MAX_SPEED_CHANGE * (time_elapsed / 0.1)  # Normalized to 100ms
            
            # Strict spike detection
            if speed_change > max_change:
                self.consecutive_spikes += 1
                print(f"Speed spike detected: {last_speed:.1f} → {speed_kmh:.1f} km/h (change: {speed_change:.1f})")
                
                # If we've had multiple consecutive spikes, use a more stable reference
                if self.consecutive_spikes > 2:
                    return self.last_stable_speed
                
                return self._get_fallback_speed()
        
        # Stage 3: Compare with median history
        self.speed_history.append(speed_kmh)
        median_speed = self._get_median(self.speed_history)
        self.speed_median_history.append(median_speed)
        
        # Reset spike counter and update stable speed
        self.consecutive_spikes = 0
        self.last_stable_speed = speed_kmh
        self.last_valid_speed = speed_kmh
        
        # Stage 4: Apply exponential smoothing
        # Use weighted average of history and new value
        smoothed_history = sum(self.speed_median_history) / len(self.speed_median_history)
        return (smoothed_history * self.HISTORY_WEIGHT) + (speed_kmh * (1.0 - self.HISTORY_WEIGHT))
    
    def _get_fallback_speed(self):
        """Get the best fallback speed value when a spike is detected"""
        # If we have enough history, use the median of recent valid speeds
        if len(self.speed_history) >= 3:
            return self._get_median(self.speed_history)
        # Otherwise use the last valid speed
        return self.last_valid_speed
    
    def _validate_rpm(self, raw_rpm):
        """Validate and smooth RPM values"""
        # Basic sanity check
        if raw_rpm < 0 or raw_rpm > 20000:
            return self.last_valid_rpm
        
        self.rpm_history.append(raw_rpm)
        self.last_valid_rpm = raw_rpm
        
        # Simple median filter
        return self._get_median(self.rpm_history)
    
    def parse_packet(self, data):
        """Parse a UDP packet from Forza Horizon 5 with strict validation"""
        try:
            # Check packet size
            if len(data) != self.packet_struct['total_size']:
                self.invalid_packet_count += 1
                if self.invalid_packet_count % 10 == 0:  # Only log every 10th invalid packet
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
            
            # Increment valid packet counter - useful for debugging
            self.valid_packet_count += 1
            
            # Extract values with multiple fallback strategies
            # RPM
            try:
                raw_rpm = all_values[self.format_definition['CurrentEngineRpm']['index']]
                rpm = self._validate_rpm(raw_rpm)
            except (IndexError, KeyError):
                rpm = self._validate_rpm(0)
            
            # Gear
            try:
                gear_value = all_values[self.format_definition['Gear']['index']]
                gear = get_gear_display(gear_value)
            except (IndexError, KeyError):
                gear = "N"  # Default to neutral
            
            # Throttle
            try:
                throttle = all_values[self.format_definition['Throttle']['index']]
                throttle = round(throttle / 255, 2) if self.format_definition['Throttle'].get('normalize') else throttle
            except (IndexError, KeyError):
                throttle = 0
            
            # Brake
            try:
                brake = all_values[self.format_definition['Brake']['index']]
                brake = round(brake / 255, 2) if self.format_definition['Brake'].get('normalize') else brake
            except (IndexError, KeyError):
                brake = 0
            
            # Calculate speed with multi-stage validation
            try:
                # First try using direct speed value (most reliable)
                speed_ms = all_values[self.format_definition['Speed']['index']]
                
                # If direct speed seems invalid, calculate from velocity components
                if speed_ms < 0 or speed_ms > 150:  # 150 m/s = ~540 km/h (reasonable max)
                    vx = all_values[self.format_definition['VelocityX']['index']]
                    vy = all_values[self.format_definition['VelocityY']['index']]
                    vz = all_values[self.format_definition['VelocityZ']['index']]
                    speed_ms = math.sqrt(vx*vx + vy*vy + vz*vz)
                
                # Apply strict validation and smoothing
                validated_speed = self._validate_speed(speed_ms)
            except (IndexError, KeyError) as e:
                print(f"Error getting speed value: {e}")
                validated_speed = self._get_fallback_speed()
            
            # Return the validated and smoothed data
            return {
                'speed': round(validated_speed, 1),
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