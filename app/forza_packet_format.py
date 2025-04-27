"""
Forza Horizon 5 Packet Format Definition

This file defines the data structure of UDP telemetry packets sent by Forza Horizon 5.
Reference: https://www.reddit.com/r/ForzaHorizon5/comments/rjqaft/telemetry_data_out_byte_ordertutorial/
"""

# Packet format for Forza Horizon 5 (324-byte format)
FH5_PACKET_FORMAT = {
    # Sled (basic simulation data)
    'IsRaceOn': {'index': 0, 'type': 'int32', 'description': 'Game is active (0 = paused or in menu)'},
    'TimestampMS': {'index': 1, 'type': 'uint32', 'description': 'Timestamp in milliseconds'},
    
    # Engine data
    'EngineMaxRpm': {'index': 2, 'type': 'float', 'description': 'Maximum engine RPM'},
    'EngineIdleRpm': {'index': 3, 'type': 'float', 'description': 'Idle engine RPM'},
    'CurrentEngineRpm': {'index': 4, 'type': 'float', 'description': 'Current engine RPM'},
    
    # Physics - acceleration (g-force)
    'AccelerationX': {'index': 5, 'type': 'float', 'description': 'Acceleration X (right)'},
    'AccelerationY': {'index': 6, 'type': 'float', 'description': 'Acceleration Y (up)'},
    'AccelerationZ': {'index': 7, 'type': 'float', 'description': 'Acceleration Z (forward)'},
    
    # Physics - velocity (m/s)
    'VelocityX': {'index': 8, 'type': 'float', 'description': 'Velocity X (right)'},
    'VelocityY': {'index': 9, 'type': 'float', 'description': 'Velocity Y (up)'},
    'VelocityZ': {'index': 10, 'type': 'float', 'description': 'Velocity Z (forward)'},
    
    # Angular velocity (rad/s)
    'AngularVelocityX': {'index': 11, 'type': 'float', 'description': 'Angular velocity X'},
    'AngularVelocityY': {'index': 12, 'type': 'float', 'description': 'Angular velocity Y'},
    'AngularVelocityZ': {'index': 13, 'type': 'float', 'description': 'Angular velocity Z'},
    
    # Car orientation (rad)
    'Yaw': {'index': 14, 'type': 'float', 'description': 'Yaw (rotation around vertical axis)'},
    'Pitch': {'index': 15, 'type': 'float', 'description': 'Pitch (rotation around lateral axis)'},
    'Roll': {'index': 16, 'type': 'float', 'description': 'Roll (rotation around longitudinal axis)'},
    
    # Wheel rotation speed (rad/s)
    'WheelRotationSpeedFrontLeft': {'index': 24, 'type': 'float', 'description': 'Front left wheel rotation speed'},
    'WheelRotationSpeedFrontRight': {'index': 25, 'type': 'float', 'description': 'Front right wheel rotation speed'},
    'WheelRotationSpeedRearLeft': {'index': 26, 'type': 'float', 'description': 'Rear left wheel rotation speed'},
    'WheelRotationSpeedRearRight': {'index': 27, 'type': 'float', 'description': 'Rear right wheel rotation speed'},
    
    # Car position (meters)
    'PositionX': {'index': 66, 'type': 'float', 'description': 'Position X in world space'},
    'PositionY': {'index': 67, 'type': 'float', 'description': 'Position Y in world space'},
    'PositionZ': {'index': 68, 'type': 'float', 'description': 'Position Z in world space'},
    
    # Speed and power
    'Speed': {'index': 69, 'type': 'float', 'description': 'Speed in meters per second'},
    'Power': {'index': 70, 'type': 'float', 'description': 'Power in watts'},
    'Torque': {'index': 71, 'type': 'float', 'description': 'Torque in newton meters'},
    
    # Tire temperatures (°F)
    'TireTemperatureFrontLeft': {'index': 72, 'type': 'float', 'description': 'Front left tire temperature'},
    'TireTemperatureFrontRight': {'index': 73, 'type': 'float', 'description': 'Front right tire temperature'},
    'TireTemperatureRearLeft': {'index': 74, 'type': 'float', 'description': 'Rear left tire temperature'},
    'TireTemperatureRearRight': {'index': 75, 'type': 'float', 'description': 'Rear right tire temperature'},
    
    # Performance metrics
    'Boost': {'index': 76, 'type': 'float', 'description': 'Boost pressure'},
    'Fuel': {'index': 77, 'type': 'float', 'description': 'Fuel remaining'},
    
    # Race data (non-float values)
    'LapNumber': {'index': 78, 'type': 'uint16', 'description': 'Current lap number'},
    'RacePosition': {'index': 79, 'type': 'uint8', 'description': 'Current race position'},
    
    # Controls (all normalized 0-1 values)
    'Throttle': {'index': 80, 'type': 'uint8', 'normalize': True, 'description': 'Throttle input (0-1)'},
    'Brake': {'index': 81, 'type': 'uint8', 'normalize': True, 'description': 'Brake input (0-1)'},
    'Clutch': {'index': 82, 'type': 'uint8', 'normalize': True, 'description': 'Clutch input (0-1)'},
    'HandBrake': {'index': 83, 'type': 'uint8', 'normalize': True, 'description': 'Handbrake input (0-1)'},
    'Gear': {'index': 84, 'type': 'uint8', 'description': 'Current gear (0=R, 1=N, 2=1st, 3=2nd, etc.)'},
    'Steer': {'index': 85, 'type': 'uint8', 'normalize': True, 'description': 'Steering input (-1 to 1)'},
}

# Packet struct layout for 324-byte Forza Horizon 5 format
FH5_PACKET_STRUCT = {
    'primitive_count': 78,           # Number of 4-byte float values (312 bytes)
    'primitive_format': '<' + 'f' * 78,  # Format string for struct.unpack
    'remaining_format': '<HB' + 'B' * 6, # Format for remaining bytes (uint16, uint8, 6 x uint8)
    'total_size': 324                # Total packet size in bytes
}

# Gear mapping for human-readable display
FH5_GEAR_MAPPING = {
    0: "R",   # Reverse
    1: "1",   # First gear
    2: "2",   # Second gear
    3: "3",   # Third gear
    4: "4",   # Fourth gear
    5: "5",   # Fifth gear
    6: "6",   # Sixth gear
    7: "7",   # Seventh gear
    8: "8",   # Eighth gear
    9: "9",  # Ninth gear
    10: "10"  # Tenth gear
}

def get_gear_display(gear_value):
    """Convert raw gear value to human-readable format"""
    return FH5_GEAR_MAPPING.get(gear_value, str(gear_value - 1))