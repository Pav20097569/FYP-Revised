# Forza Dashboard with CANoe Integration

A real-time telemetry dashboard that integrates Forza Horizon 5 with Vector CANoe for automotive diagnostics and machine learning-based analysis.

## Overview

This project creates a bridge between the realistic driving simulation of Forza Horizon 5 and professional automotive diagnostic tools. It features:

- Real-time telemetry visualization from Forza Horizon 5
- Integration with Vector CANoe for DTC (Diagnostic Trouble Code) management
- Machine learning-based anomaly detection and DTC prediction
- Root cause analysis of driving patterns and DTCs
- Comprehensive data logging and export for offline analysis

## Features

- **Live Dashboard**: Real-time visualization of vehicle telemetry with smoothing algorithms
- **CANoe Integration**: Capture and analyze DTCs from Vector CANoe
- **AI Analysis**: Machine learning models to detect anomalies and predict potential DTCs
- **Root Cause Analysis**: Statistical correlation between driving patterns and specific DTCs
- **Data Export**: Export telemetry and analysis data for offline processing

## System Requirements

- Windows 10/11 (for Forza Horizon 5 and CANoe)
- Python 3.8 or higher
- Forza Horizon 5 with Data Out enabled
- Vector CANoe (optional, system will work in fallback mode without it)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation

1. Clone the repository
2. Install requirements:
```
pip install -r requirements.txt
```
3. Configure Forza Horizon 5 Data Out to send UDP data to 127.0.0.1:5300
4. (Optional) Set CANOE_CONFIG_PATH environment variable to point to your CANoe configuration

## Usage

Start the server:
```
python main.py
```

Then open a web browser and navigate to:
```
http://127.0.0.1:5300
```

## Project Structure

```
├── analysis_results/      # DTC analysis results and visualizations
├── app/                   # Main application package
│   ├── __pycache__/       # Python cache directory
│   ├── __init__.py        # App initialization
│   ├── canoe_integration.py  # CANoe integration module
│   ├── data_learning.py   # Machine learning components
│   ├── data_logger.py     # Telemetry logging module
│   ├── data_parser.py     # Telemetry data parsing
│   ├── forza_packet_format.py  # Forza telemetry packet definitions
│   ├── telemetry_listener.py  # UDP listener for Forza telemetry
│   └── websocket_handler.py  # WebSocket event handlers
├── dtc_logs/              # Logs of DTC events
├── models/                # Saved machine learning models
├── telemetry_exports/     # Exported telemetry data in CSV format
├── templates/             # HTML templates for web interface
│   ├── index.html         # Main dashboard HTML
│   └── model_dashboard.html  # Alternative dashboard visualization
├── main.py                # Application entry point
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

Additional files:
```
├── check_project.py       # Project verification script
├── fix_json.py            # JSON utility script
├── model_analysis.py      # Model analysis utilities
├── model_dashboard.py     # Dashboard model handler
├── scriptForDTC.py        # DTC script utilities
└── test_s3_connection.py  # S3 connection test utility
```

## Data Flow

1. Forza Horizon 5 sends UDP telemetry data
2. The telemetry listener receives and validates the data
3. Data is processed, logged, and distributed via WebSockets
4. The dashboard displays real-time telemetry
5. Machine learning models analyze telemetry for anomalies and DTC predictions
6. CANoe integration captures and processes DTCs
7. Root cause analysis identifies correlations between driving patterns and DTCs

## Machine Learning Components

The system includes several machine learning components:

- **Anomaly Detection**: Isolation Forest algorithm to detect unusual driving patterns
- **DTC Prediction**: Random Forest classifiers to predict potential DTCs
- **Root Cause Analysis**: Statistical correlation and effect size calculations
- **Feature Engineering**: Time-window based feature extraction from telemetry data

## API Documentation

### WebSocket Events

#### Client to Server:
- `connect`: Connect to the server
- `disconnect`: Disconnect from the server
- `export_telemetry`: Request telemetry export to CSV
- `clear_dtcs`: Request clearing of all DTCs
- `add_test_dtc`: Add a test DTC with specified parameters

#### Server to Client:
- `telemetry`: Real-time telemetry updates
- `dtc_update`: Current state of all DTCs
- `dtc_new`: Notification of new DTCs
- `dtc_cleared`: Notification of cleared DTCs
- `telemetry_analysis`: Results of AI analysis
- `training_complete`: Notification of completed model training

### REST API

- `GET /api/dtcs`: Get current active DTCs
- `GET /api/dtc-analysis`: Get DTC root cause analysis
- `GET /api/train-models`: Trigger background training of ML models

## License

[MIT License](LICENSE)

## Acknowledgments

- The Forza Horizon 5 team for providing telemetry data output
- Vector for CANoe diagnostic tools