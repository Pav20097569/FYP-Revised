import eventlet
eventlet.monkey_patch()

from app import create_app, socketio
from app.cloud_upload import register_s3_auto_upload  # Import the auto-upload module

app = create_app()

# Register S3 auto-upload BEFORE running the app
register_s3_auto_upload(
    bucket_name="telemetrydatapav",  # Bucket name
    region="us-east-1",              # Region
    source_dir="analysis_results"    # The directory to upload
)

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5300)