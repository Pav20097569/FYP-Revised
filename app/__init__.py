from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from .telemetry_listener import start_listener
import os

# Initialize Socket.IO with CORS enabled for all origins
socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    # Get the absolute path to the template folder
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
    
    # Create Flask app with explicit template folder
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    socketio.init_app(app)
    
    # Register WebSocket handlers
    from .websocket_handler import register_handlers
    register_handlers(socketio)
    
    # Serve the main dashboard
    @app.route('/')
    def index():
        return send_from_directory(template_dir, 'index.html')
    
    # Start the UDP telemetry listener
    start_listener(socketio)
    
    return app