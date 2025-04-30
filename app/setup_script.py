"""
Setup Script for Forza Dashboard

This script ensures the correct file structure and moves index.html to the templates directory.
"""

import os
import shutil
import sys

def setup_project():
    """Set up the project directories and move files to the right locations"""
    # Create necessary directories
    directories = [
        'templates',
        'static',
        'telemetry_exports',
        'dtc_logs',
        'models',
        'analysis_results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Move index.html to templates directory if it exists in root
    if os.path.exists('index.html') and not os.path.exists(os.path.join('templates', 'index.html')):
        shutil.copy('index.html', os.path.join('templates', 'index.html'))
        print("Copied index.html to templates directory")
    
    print("Setup complete!")

if __name__ == "__main__":
    setup_project()