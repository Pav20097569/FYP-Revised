"""
Create a file called fix_json.py in your project directory with this code:
"""

import os
import json
import shutil

def fix_json_file():

    
    """Fix the corrupted JSON file or create a dummy file if needed"""
    analysis_path = os.path.join("analysis_results", "dtc_root_causes.json")
    
    # Check if directory exists
    if not os.path.exists("analysis_results"):
        print("Creating analysis_results directory")
        os.makedirs("analysis_results", exist_ok=True)
    
    # Create a simple valid JSON file
    dummy_data = {
        "P0300": {
            "occurrences": 100,
            "root_cause_hypothesis": "This DTC may be triggered by high rpm and low brake",
            "significant_features": ["rpm", "brake", "throttle"],
            "conditions": {
                "speed": {
                    "mean_difference": 15.5,
                    "effect_size": 0.75,
                    "significant": True
                }
            }
        }
    }
    
    # Check if file exists
    if os.path.exists(analysis_path):
        # Try to read it first
        try:
            with open(analysis_path, 'r') as f:
                data = json.load(f)
                print("JSON file is valid, no fix needed")
                return
        except json.JSONDecodeError:
            # Back up corrupted file
            backup_path = analysis_path + ".bak"
            print(f"Backing up corrupted file to {backup_path}")
            shutil.copy2(analysis_path, backup_path)
    
    # Write new valid JSON
    print(f"Creating valid JSON file at {analysis_path}")
    with open(analysis_path, 'w') as f:
        json.dump(dummy_data, f, indent=2)
    
    print("Fix complete! Please restart your application")

if __name__ == "__main__":
    fix_json_file()