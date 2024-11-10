import json
import os
from typing import List, Dict, Any
import hashlib
import pandas as pd
from scipy import stats
import numpy as np
from itertools import combinations

def hash_dict(d: Dict[str, Any]) -> str:
    """Create a hash for a dictionary."""
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()

def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_experiment_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract relevant data from all experiments in the dictionary."""
    experiments = []
    for exp_id, exp_data in data['experiments'].items():
        experiment = exp_data['experiment']
        experiments.append({
            'experiment_id': exp_id,
            'totalFunctionEvaluations': experiment['totalFunctionEvaluations'],
            'classifierModel': experiment['classifierModel'],
            'datasetName': experiment['datasetName'],
            'solver': experiment['solver'],
            'hyperParameters_hash': hash_dict(experiment['hyperParameters']),
            'accuracies': [solution['bestAccuracy'] for solution in experiment['solutions']]
        })
    return experiments

def create_dataframe(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a pandas DataFrame from the list of experiments."""
    df = pd.DataFrame(experiments)
    df = df.explode('accuracies').reset_index(drop=True)
    df['accuracies'] = df['accuracies'].astype(float)
    return df

def group_experiments(df: pd.DataFrame):
    """Group experiments based on specified fields."""
    return df.groupby(['totalFunctionEvaluations', 'classifierModel', 
                       'datasetName', 'hyperParameters_hash'])
def create_comparison_matrix(group):
    """Create a comparison matrix for a group of experiments."""
    experiment_ids = sorted(group['experiment_id'].unique())
    n = len(experiment_ids)
    matrix = np.full((n, n), 0.5)  # 0.5 indicates equality (diagonal)
    
    for i, exp1 in enumerate(experiment_ids):
        for j, exp2 in enumerate(experiment_ids):
            if i != j:
                accuracies1 = group[group['experiment_id'] == exp1]['accuracies']
                accuracies2 = group[group['experiment_id'] == exp2]['accuracies']
                
                statistic, p_value = stats.ranksums(accuracies1.array, accuracies2.array)
                
                if p_value < 0.05:
                    # 1.0 means exp1 is better, 0.0 means exp2 is better
                    matrix[i, j] = 1.0 if statistic > 0 else 0.0
                else:
                    # 0.5 means no significant difference
                    matrix[i, j] = 0.5
    
    return matrix

def compare_experiments(grouped_df):
    """Compare experiments and generate matrices for each group."""
    matrices = {}
    
    for name, group in grouped_df:
        if len(group['experiment_id'].unique()) < 2:
            continue
            
        # Create a descriptive key for the matrix
        key = f"{name[1]}-{name[2]}"  # classifierModel-datasetName
        matrices[key] = create_comparison_matrix(group)
        
        # Print textual results for verification
        print(f"\nComparison matrix for {key}:")
        print(matrices[key])
    
    return matrices

def main(file_paths: List[str]):
    all_experiments = []
    for file_path in file_paths:
        data = read_json_file(file_path)
        experiments = extract_experiment_data(data)
        all_experiments.extend(experiments)

    df = create_dataframe(all_experiments)
    grouped_df = group_experiments(df)
    matrices = compare_experiments(grouped_df)
    
    # Save matrices to JSON for visualization
    with open('comparison_matrices.json', 'w') as f:
        json.dump({k: m.tolist() for k, m in matrices.items()}, f)


if __name__ == "__main__":
    # Replace with your actual file paths
    LOGS_ROOT="/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs"
    problemCategories=["classification"]
    models=["RandomForest","SVM","GradientBoost","DecisionTree"]
    # experiments=["smallDatasets/smallIter","smallDatasets/HybridBayes"]
    # experiments=["smallDatasets/HybridBayes"]
    experiments=["smallDatasets/smallIter"]
    solvers=["pyNMHH","bayesGP"]
    for problemCategory in problemCategories:
            for experiment in experiments:
                for model in models:
                    file_paths = [f"{LOGS_ROOT}/{solver}/{problemCategory}/{model}/{experiment}/records.json" for solver in solvers]
                    print(f"Category: {problemCategory}, model: {model}, experiment: {experiment}, solvers: {solvers}")
                    main(file_paths)