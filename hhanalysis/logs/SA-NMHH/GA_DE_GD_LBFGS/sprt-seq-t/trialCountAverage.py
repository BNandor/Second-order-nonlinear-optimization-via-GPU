import os
import json

def calculate_average_trial_count(directory_path):
    trial_counts = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json") and filename != "records.json":
            file_path = os.path.join(directory_path, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Assuming /experiments is an array of objects in the JSON data
                experiments = data.get("experiments", [])
                for experiment in experiments:
                    
                    trial_count = experiment["trialCount"]
                    if trial_count is not None:
                        trial_counts.append(trial_count)
    
    if trial_counts:
        average_trial_count = sum(trial_counts) / len(trial_counts)
        return average_trial_count
    else:
        return None

# Example usage
directory_path = './'
average = calculate_average_trial_count(directory_path)
if average is not None:
    print(f"Average trialCount: {average}")
else:
    print("No trialCount values found.")
