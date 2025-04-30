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
        self.speed_history = deque(maxlen=15)  # Increased buffer size
        self.speed_median_history = deque(maxlen=8)  # For median filtering
        self.last_valid_speed = 0.0
        self.last_update_time = time.time()
        
        # Updated limits with more lenient parameters
        self.MAX_SPEED_CHANGE = 25.0  # More lenient limit (km/h per update)
        self.MAX_REASONABLE_SPEED = 600.0  # Much higher max speed
        self.MIN_REASONABLE_SPEED = -5.0  # Allow slightly negative values
        
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
        
        # Debug settings
        self.debug_mode = False  # Set to True to enable verbose logging
        self.last_log_time = 0
        self.log_interval = 5.0  # Only log every 5 seconds in non-debug mode
        
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
        """Improved speed validation and smoothing to handle extreme values better"""
        # Convert to km/h
        speed_kmh = raw_speed * 3.6
        
        # Stage 1: Basic range check with more lenient limits
        if speed_kmh < -5.0 or speed_kmh > 600.0:  # Much more lenient upper limit
            self._log_speed_issue(f"Speed extremely out of range: {speed_kmh:.1f} km/h - treating as game menu/reset")
            # When this happens, it's likely the game is in a menu or has just reset
            # Instead of printing a warning each time, we'll silently return last valid speed
            return self.last_valid_speed
        
        # Stage 2: Speed change check with game state awareness
        if self.speed_history:
            last_speed = self.speed_history[-1]
            speed_change = abs(speed_kmh - last_speed)
            
            current_time = time.time()
            time_elapsed = max(0.001, current_time - self.last_update_time)  # Avoid division by zero
            self.last_update_time = current_time
            
            # Calculate maximum allowed change based on time elapsed
            max_change = self.MAX_SPEED_CHANGE * (time_elapsed / 0.1)  # Normalized to 100ms
            
            # For very high speeds, allow larger changes
            if last_speed > 200:
                max_change *= 2.0  # Double the allowed change at high speeds
            
            # Special case: If current or last speed is near-zero, we're starting/stopping
            is_starting_or_stopping = (last_speed < 5.0 or speed_kmh < 5.0)
            
            # Spike detection with game state awareness
            if speed_change > max_change and not is_starting_or_stopping:
                self.consecutive_spikes += 1
                
                # Only print a message every few spikes to avoid log flooding
                if self.consecutive_spikes % 5 == 1:
                    self._log_speed_issue(f"Speed spike detected: {last_speed:.1f} → {speed_kmh:.1f} km/h (change: {speed_change:.1f})")
                
                # If we've had multiple consecutive spikes, use a more stable reference
                # Apply gradual transition instead of rejecting the value entirely
                if self.consecutive_spikes > 3:
                    # Blend between history and new value with more weight to history
                    blend_ratio = 0.9  # 90% history, 10% new value
                    return (self.last_stable_speed * blend_ratio) + (speed_kmh * (1.0 - blend_ratio))
                
                return self._get_fallback_speed()
        
        # Stage 3: Reset spike counter and update history
        self.consecutive_spikes = 0
        self.speed_history.append(speed_kmh)
        median_speed = self._get_median(self.speed_history)
        self.speed_median_history.append(median_speed)
        
        # Update reference values
        self.last_stable_speed = speed_kmh
        self.last_valid_speed = speed_kmh
        
        # Stage 4: Apply exponential smoothing - weighted average of history and new value
        # Use stronger smoothing for high speeds to reduce jitter
        if speed_kmh > 200:
            history_weight = min(0.92, self.HISTORY_WEIGHT + 0.07)  # Increase smoothing for high speeds
        else:
            history_weight = self.HISTORY_WEIGHT
        
        smoothed_history = sum(self.speed_median_history) / len(self.speed_median_history)
        return (smoothed_history * history_weight) + (speed_kmh * (1.0 - history_weight))
    
    def _validate_rpm(self, raw_rpm):
        """Validate and smooth RPM values"""
        # Basic sanity check
        if raw_rpm < 0 or raw_rpm > 20000:
            return self.last_valid_rpm
        
        self.rpm_history.append(raw_rpm)
        self.last_valid_rpm = raw_rpm
        
        # Simple median filter
        return self._get_median(self.rpm_history)
    
    def _get_fallback_speed(self):
        """Get the best fallback speed value when a spike is detected using multiple strategies"""
        # Strategy 1: Use median of recent valid speeds if we have enough history
        if len(self.speed_history) >= 3:
            median_speed = self._get_median(self.speed_history)
            # If we have a very recent last_valid_speed, blend it with the median
            if hasattr(self, 'last_update_time') and time.time() - self.last_update_time < 0.5:
                return (median_speed * 0.7) + (self.last_valid_speed * 0.3)
            return median_speed
        
        # Strategy 2: Use exponential weighted average if we have some history
        if len(self.speed_history) > 0:
            # Calculate exponentially weighted average (more weight to recent values)
            weights = [0.6 ** i for i in range(len(self.speed_history))]
            weight_sum = sum(weights)
            if weight_sum > 0:
                weighted_speed = sum(s * w for s, w in zip(reversed(self.speed_history), weights)) / weight_sum
                return weighted_speed
        
        # Strategy 3: As last resort, use the last valid speed
        return self.last_valid_speed
    
    def _log_speed_issue(self, message):
        """Log speed issues with rate limiting to avoid flooding the console"""
        current_time = time.time()
        
        # Always log in debug mode
        if self.debug_mode:
            print(message)
            return
        
        # Otherwise, rate limit messages
        if current_time - self.last_log_time > self.log_interval:
            print(message)
            self.last_log_time = current_time
    
    def parse_packet(self, data):
        """Parse a UDP packet from Forza Horizon 5 with improved validation"""
        try:
            # Check packet size
            if len(data) != self.packet_struct['total_size']:
                self.invalid_packet_count += 1
                if self.invalid_packet_count % 10 == 0:  # Only log every 10th invalid packet
                    self._log_speed_issue(f"Unexpected packet size: {len(data)} bytes (expected {self.packet_struct['total_size']})")
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
            is_race_on = all_values[self.format_definition['IsRaceOn']['index']]
            if is_race_on == 0:
                # Game is paused or in a menu - return last valid data with reduced speed
                if hasattr(self, 'last_valid_result') and self.last_valid_result:
                    # Create a copy with gradually decreasing speed
                    result = self.last_valid_result.copy()
                    result['speed'] = max(0, result['speed'] * 0.7)  # Decay speed by 30%
                    return result
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
                if speed_ms < -10 or speed_ms > 150:  # 150 m/s = ~540 km/h (reasonable max)
                    vx = all_values[self.format_definition['VelocityX']['index']]
                    vy = all_values[self.format_definition['VelocityY']['index']]
                    vz = all_values[self.format_definition['VelocityZ']['index']]
                    speed_ms = math.sqrt(vx*vx + vy*vy + vz*vz)
                
                # Apply improved validation and smoothing
                validated_speed = self._validate_speed(speed_ms)
            except (IndexError, KeyError) as e:
                if self.debug_mode:
                    print(f"Error getting speed value: {e}")
                validated_speed = self._get_fallback_speed()
            
            # Assemble the final result
            result = {
                'speed': round(validated_speed, 1),
                'rpm': round(rpm),
                'gear': gear,
                'throttle': throttle,
                'brake': brake
            }
            
            # Store this as last valid result
            self.last_valid_result = result
            
            return result
        
        except Exception as e:
            self._log_speed_issue(f"Error parsing Forza data: {e}")
            return None

# Create a singleton instance
parser = ForzaDataParser()

def parse_packet(data):
    """Helper function that uses the singleton parser"""
    return parser.parse_packet(data)