# Forza Telemetry Dashboard

A real-time web dashboard for displaying Forza Horizon 5 telemetry data.

## Features

- Real-time display of speed, RPM, gear, throttle and brake inputs
- Smooth animations and transitions
- Responsive design that works on all devices
- Automatic connection status indicator

## Requirements

- Python 3.7 or higher
- Forza Horizon 5 game
- Flask and Flask-SocketIO libraries

## Installation

1. Clone this repository or download the files
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the setup script to create necessary directories:

```bash
python setup.py
```

## Configuration in Forza Horizon 5

1. Launch Forza Horizon 5
2. Go to Settings > HUD and Gameplay
3. Find "Data Out" settings
4. Set:
   - Data Out: ON
   - Data Out IP Address: your computer's IP address where this server is running
   - Data Out IP Port: 5300
   - Data Out Packet Format: 324 bytes
   - Data Out Car Indexes: 0-1000

## Running the Dashboard

Start the server:

```bash
python main.py
```

Then open a web browser and navigate to:
```
http://localhost:5300
```

## Troubleshooting

### Not receiving telemetry data?

1. Make sure Forza Horizon 5 "Data Out" feature is properly configured
2. Check if your firewall is blocking UDP port 5300
3. Verify that your computer's IP address is correctly entered in Forza settings
4. Try running the game and server on the same machine first

### Connection issues?

- The dashboard will show "Disconnected" if the connection to the server is lost
- Check that the server is still running
- Reload the page to attempt reconnection

## File Structure

- `main.py` - Entry point for the application
- `app/__init__.py` - Flask application setup
- `app/data_parser.py` - Parses Forza telemetry data
- `app/telemetry_listener.py` - Listens for UDP packets from Forza
- `app/websocket_handler.py` - Handles WebSocket connections
- `templates/index.html` - Dashboard HTML interface
- `static/style.css` - Dashboard styling

## How the Data Parser Works

The data parser interprets the 324-byte UDP packets sent by Forza Horizon 5 according to the official data format. Each packet contains various vehicle telemetry values including:

- Speed (in m/s, converted to km/h)
- Engine RPM
- Current gear
- Throttle and brake positions
- And many more values

The parser extracts these values and passes them to the web interface via WebSockets for real-time display.