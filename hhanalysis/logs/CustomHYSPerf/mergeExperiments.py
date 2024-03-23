import os
import json
def defaultTo(exp,key,val):
    if key not in exp:
        exp[key]=val
def merge_json_files(base_json_path, customhys_paths):
    # Load the base JSON file
    with open(base_json_path, 'r') as base_file:
        base_data = json.load(base_file)

    # Loop through customhys files and update base_data
    for customhys_path in customhys_paths:
        with open(customhys_path, 'r') as customhys_file:
            customhys_data = json.load(customhys_file)

        # Update base_data with experiments not already present
        for experiment_name, experiment_data in customhys_data.get("experiments", {}).items():
            defaultTo(experiment_data,"hhsteps",100)
            defaultTo(experiment_data,"baselevelIterations", 100)
            defaultTo(experiment_data,"populationSize", 30)
            defaultTo(experiment_data,"trialSampleSizes", 30)
            if experiment_name not in base_data["experiments"]:
                base_data["experiments"][experiment_name] = experiment_data

    # Write the updated data back to the base JSON file
    with open(base_json_path, 'w') as base_file:
        json.dump(base_data, base_file, indent=2)

if __name__ == "__main__":
    # Specify the base JSON file and subdirectories
    base_json_path = "allRecords.json"
    subdirectories = ["michalewiczDixonPriceLecy750/", "schwefelSquaresSphere750/", "tridSchwefel223Qing750/","rosenbrockRastriginStyblinskitang750/"]  # Add your subdirectories here

    # Get the full paths of customhys JSON files in subdirectories
    customhys_paths = []
    for subdir in subdirectories:
        customhys_path = os.path.join(subdir, "customhys.json")
        if os.path.exists(customhys_path):
            customhys_paths.append(customhys_path)

    # Merge customhys JSON files into allRecords.json
    merge_json_files(base_json_path, customhys_paths)

    print("Merging completed successfully.")
