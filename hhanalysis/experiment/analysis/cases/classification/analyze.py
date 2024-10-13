import json
import os
from typing import List, Dict, Any
import hashlib
import pandas as pd
from scipy import stats
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

def compare_experiments(grouped_df):
    """Compare experiments using Wilcoxon rank-sum test."""
    for name, group in grouped_df:
        print(f"\nComparing experiments for: {name}")
        if len(group['experiment_id'].unique()) < 2:
            print("Not enough experiments to compare.")
            continue

        experiment_ids = group['experiment_id'].unique()
        for (exp1, exp2) in combinations(experiment_ids, 2):
            accuracies1 = group[group['experiment_id'] == exp1]['accuracies']
            accuracies2 = group[group['experiment_id'] == exp2]['accuracies']
            
            statistic, p_value = stats.ranksums(accuracies1.array, accuracies2.array)
            # (statistic,pvalue)=sp.stats.ranksums(samples[i], samples[j], alternative='less')
            # (statistic,pvalue)=sp.stats.ranksums(samples[i], samples[j])
            # comparisonMatrix[i][j]=1-(pvalue<0.05)
            print(f"Wilcoxon rank-sum test between {exp1} and {exp2}:")
            print(f"Statistic: {statistic}")
            print(f"p-value: {p_value}")
            
            if p_value < 0.05:
                if statistic > 0:
                    print(f"Experiment {exp1} is significantly more accurate than Experiment {exp2}")
                else:
                    print(f"Experiment {exp2} is significantly more accurate than Experiment {exp1}")
            else:
                print(f"No significant difference in accuracy between Experiments {exp1} and {exp2}")

def main(file_paths: List[str]):
    all_experiments = []
    for file_path in file_paths:
        data = read_json_file(file_path)
        experiments = extract_experiment_data(data)
        all_experiments.extend(experiments)

    df = create_dataframe(all_experiments)
    grouped_df = group_experiments(df)
    compare_experiments(grouped_df)

if __name__ == "__main__":
    # Replace with your actual file paths
    file_paths = ["/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs/pyNMHH/classification/RandomForest/smallDatasets/smallIter/records.json",
                  "/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs/bayesGP/classification/RandomForest/smallDatasets/smallIter/records.json"]  # Add more file paths as needed
    main(file_paths)