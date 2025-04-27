import os
import sys

def check_project_structure():
    """
    Check if the Forza Dashboard project structure is correct
    """
    print("Checking Forza Dashboard project structure...")
    
    # Required files and directories
    required_files = [
        "main.py",
        "setup.py",
        "requirements.txt",
        "app/__init__.py",
        "app/data_parser.py",
        "app/telemetry_listener.py",
        "app/websocket_handler.py",
        "templates/index.html"
    ]
    
    missing_files = []
    
    # Check if files exist
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nYou need to create the following directories and files:")
        print("1. Make sure you have an 'app' directory in your project root")
        print("2. Place the provided Python files in the app directory")
        print("3. Run setup.py to create the templates directory and HTML file")
    else:
        print("✓ All required files found!")
        print("\nYour project structure is correct. You can run the application with:")
        print("  python main.py")
    
    # Check app directory structure
    if not os.path.exists("app"):
        print("\n❌ 'app' directory is missing. Create it in your project root.")
    elif not os.path.isdir("app"):
        print("\n❌ 'app' exists but is not a directory. It should be a directory.")
    
    # Check templates directory
    if not os.path.exists("templates"):
        print("\n❌ 'templates' directory is missing. Run setup.py to create it.")
    elif not os.path.isdir("templates"):
        print("\n❌ 'templates' exists but is not a directory. It should be a directory.")

if __name__ == "__main__":
    check_project_structure()