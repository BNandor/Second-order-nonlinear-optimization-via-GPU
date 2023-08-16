import json
import pandas as pd
import sys

script_filename = sys.argv[1]

# Load the main experiments JSON file into a DataFrame
with open(script_filename, 'r') as main_file:
    main_data = json.load(main_file)

dfs = []

# Iterate through each experiment in the main file
for experiment_hash, experiment_data in main_data['experiments'].items():
    experiment_data_flat = {f'experiment.{k}': v for k, v in experiment_data['experiment'].items()}
    metadata_data_flat = {f'metadata.{k}': v for k, v in experiment_data['metadata'].items()}
    combined_data = {**experiment_data_flat, **metadata_data_flat}
    df = pd.DataFrame([combined_data])  # Create a DataFrame with a single row
    dfs.append(df)

# Concatenate the DataFrames
df = pd.concat(dfs, ignore_index=True)
# Extract relevant columns for analysis
df['Problem'] = df['experiment.problems'].apply(lambda x: x[0])
df['Dimensionality'] = df['experiment.modelSize']
df['Elapsed Time (s)'] = df['metadata.elapsedTimeSec'].astype(float)

# Perform data analysis
average_time_per_problem = df.groupby('Problem')['Elapsed Time (s)'].mean()
max_time_per_dimensionality = df.groupby('Dimensionality')['Elapsed Time (s)'].max()

# Print the analysis results
print("Average Elapsed Time per Problem:")
print(average_time_per_problem)
print("\nMax Elapsed Time per Dimensionality:")
print(max_time_per_dimensionality)

