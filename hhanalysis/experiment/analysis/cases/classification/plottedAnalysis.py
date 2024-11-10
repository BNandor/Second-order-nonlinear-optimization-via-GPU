import json
import os
from typing import List, Dict, Any
import hashlib
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

def hash_dict(d: Dict[str, Any]) -> str:
    """Create a hash for a dictionary."""
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def create_WilcoxonRanksum_matrix(group):
    """Create a comparison matrix for a group of experiments."""
    solvers = sorted(group['solver'].unique())
    n = len(solvers)
    matrix = np.full((n, n), 0.5)  # 0.5 indicates equality (diagonal)
    
    for i, solver1 in enumerate(solvers):
        for j, solver2 in enumerate(solvers):
            if i != j:
                accuracies1 = group[group['solver'] == solver1]['accuracies']
                accuracies2 = group[group['solver'] == solver2]['accuracies']
                if(len(accuracies1.array)!=len(accuracies2.array)):
                    print("Invalid comparison, check the experiment groupings!")
                    exit
                statistic, p_value = stats.ranksums(accuracies1.array, accuracies2.array)
                
                if p_value < 0.05:
                    # 1.0 means solver1 is better, 0.0 means solver2 is better
                    matrix[i, j] = 1.0 if statistic > 0 else 0.0
                else:
                    # 0.5 means no significant difference
                    matrix[i, j] = 0.5
    
    return matrix, solvers

def create_medIQR_matrix(group):
    """Create a comparison matrix for a group of experiments."""
    solvers = sorted(group['solver'].unique())
    n = len(solvers)
    matrix = np.full((n, n), 0.5)  # 0.5 indicates equality (diagonal)
    
    for i, solver1 in enumerate(solvers):
        for j, solver2 in enumerate(solvers):
            if i != j:
                accuracies1 = group[group['solver'] == solver1]['accuracies']
                accuracies2 = group[group['solver'] == solver2]['accuracies']
                
                statistic, p_value = stats.ranksums(accuracies1.array, accuracies2.array)
                medIQR1=np.median(accuracies1.array)-stats.iqr(accuracies1.array)
                medIQR2=np.median(accuracies2.array)-stats.iqr(accuracies2.array)
                if medIQR1-medIQR2>0.000001:
                    # 1.0 means solver1 is better, 0.0 means solver2 is better
                    matrix[i, j] = 1.0 
                elif abs(medIQR1-medIQR2)<0.000001:
                    # 0.5 means no significant difference
                    matrix[i, j] = 0.5
                else: 
                    matrix[i, j] = 0.0
    
    return matrix, solvers

def create_mean_matrix(group):
    """Create a comparison matrix for a group of experiments."""
    solvers = sorted(group['solver'].unique())
    n = len(solvers)
    matrix = np.full((n, n), 0.5)  # 0.5 indicates equality (diagonal)
    
    for i, solver1 in enumerate(solvers):
        for j, solver2 in enumerate(solvers):
            if i != j:
                accuracies1 = group[group['solver'] == solver1]['accuracies']
                accuracies2 = group[group['solver'] == solver2]['accuracies']
                
                statistic, p_value = stats.ranksums(accuracies1.array, accuracies2.array)
                mean1=np.mean(accuracies1.array)#-stats.iqr(accuracies1.array)
                mean2=np.mean(accuracies2.array)#-stats.iqr(accuracies2.array)
                if mean1-mean2>0.000001:
                    # 1.0 means solver1 is better, 0.0 means solver2 is better
                    matrix[i, j] = 1.0 
                elif abs(mean1-mean2)<0.000001:
                    # 0.5 means no significant difference
                    matrix[i, j] = 0.5
                else: 
                    matrix[i, j] = 0.0
    
    return matrix, solvers

def create_maxcomparison_matrix(group):
    """Create a comparison matrix for a group of experiments."""
    solvers = sorted(group['solver'].unique())
    n = len(solvers)
    matrix = np.full((n, n), 0.5)  # 0.5 indicates equality (diagonal)
    
    for i, solver1 in enumerate(solvers):
        for j, solver2 in enumerate(solvers):
            if i != j:
                accuracies1 = group[group['solver'] == solver1]['accuracies']
                accuracies2 = group[group['solver'] == solver2]['accuracies']
                # if(len(accuracies1.array)!=len(accuracies2.array)):
                #     print("Invalid comparison, check the experiment groupings!")
                #     exit
                max1=np.max(accuracies1)
                max2=np.max(accuracies2)
                statistic, p_value = stats.ranksums(accuracies1.array, accuracies2.array)
                
                if max1-max2>0.0001:
                    matrix[i, j] = 1.0 
                elif abs(max1-max2)<0.0001:
                    # 0.5 means no significant difference
                    matrix[i, j] = 0.5
                else: 
                    matrix[i, j] = 0.0

    return matrix, solvers

def getSolverTimes(group):
    solvers = sorted(group['solver'].unique())
    allSolverTimes={}                
    for i, solver in enumerate(solvers):
        solverTimes=group[group['solver'] == solver]['timeSec'].array
        allSolverTimes[solver]=solverTimes[0]/len(solverTimes)
    return allSolverTimes

def group_experiments(df: pd.DataFrame):
    """Group experiments based on specified fields."""
    # return df.groupby(['totalFunctionEvaluations', 'classifierModel', 
    #                    'datasetName', 'hyperParameters_hash'])
    return df.groupby(['classifierModel','datasetName','hyperParameters_hash'])

# def plot_comparison_matrices(df):
#     """Plot comparison matrices organized by classifier (rows) and dataset (columns)."""
#     # Get unique values for grid organization
#     classifiers = sorted(df['classifierModel'].unique())
#     datasets = sorted(df['datasetName'].unique())
    
#     # Create figure
#     n_rows = len(classifiers)
#     n_cols = len(datasets)
#     fig, axes = plt.subplots(n_rows, n_cols, 
#                             figsize=(5 * n_cols, 5 * n_rows),
#                             squeeze=False)
    
#     # Create custom colormap: red -> white -> blue
#     colors = ['#ff0000', '#ffffff', '#0000ff']
#     custom_cmap = sns.blend_palette(colors, 256, as_cmap=True)
    
#     # Plot each subplot
#     for i, classifier in enumerate(classifiers):
#         for j, dataset in enumerate(datasets):
#             # Filter data for this classifier-dataset combination
#             subset = df[(df['classifierModel'] == classifier) & 
#                        (df['datasetName'] == dataset)]
            
#             if not subset.empty:
#                 # Create matrix and get solver names
#                 matrix, solvers = create_maxcomparison_matrix(subset)
#                 times=getSolverTimes(subset)
#                 # Plot heatmap
#                 sns.heatmap(matrix,
#                            annot=True,
#                            fmt='.2f',
#                            cmap=custom_cmap,
#                            center=0.5,
#                            vmin=0,
#                            vmax=1,
#                            square=True,
#                            xticklabels=False,
#                            yticklabels=solvers,
#                            ax=axes[i, j],
#                            cbar_kws={'label': 'Comparison Result'},
#                            cbar=False)
#                 # Set titles and labels
#                 # if i == 0:  # Only top row gets column titles
#                 axes[i , j].set_title(f'{dataset}')
#                 # if j == 0:  # Only first column gets row titles
#                 axes[i, j].set_ylabel(f'{classifier}')
#                 axes[i, j].tick_params(left=True, bottom=False) 
#                 # Rotate labels
#                 # plt.setp(axes[i, j].get_xticklabels(), rotation=, ha='right')
#                 plt.setp(axes[i, j].get_yticklabels(), rotation=0)
#             else:
#                 axes[i, j].axis('off')
    
#     # Add overall title
#     plt.suptitle('\n\n', fontsize=16, y=1.02)
    
#     # Adjust layout
#     plt.tight_layout()

    # # Show plot
    # plt.show()
def plot_comparison_matrices(df):
    """Plot comparison matrices with time bar plots organized by classifier (rows) and dataset (columns)."""
    # Get unique values for grid organization
    classifiers = sorted(df['classifierModel'].unique())
    datasets = sorted(df['datasetName'].unique())
    
    # Create figure - note the gridspec to handle the subplot layout
    n_rows = len(classifiers)
    n_cols = len(datasets)
    fig = plt.figure(figsize=(8 * n_cols, 5 * n_rows))
    
    # Create a grid with proper spacing
    outer_grid = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.4)
    
    # Create custom colormap: red -> white -> blue
    colors = ['#ff0000', '#ffffff', '#0000ff']
    custom_cmap = sns.blend_palette(colors, 256, as_cmap=True)
    
    # Plot each subplot
    for i, classifier in enumerate(classifiers):
        for j, dataset in enumerate(datasets):
            # Filter data for this classifier-dataset combination
            subset = df[(df['classifierModel'] == classifier) &
                       (df['datasetName'] == dataset)]
            
            if not subset.empty:
                # Create inner gridspec for matrix and barplot
                inner_grid = outer_grid[i, j].subgridspec(1, 2, width_ratios=[1.5, 1])
                
                # Create matrix and get solver names and times
                matrix, solvers = create_maxcomparison_matrix(subset)
                times = getSolverTimes(subset)
                
                # Plot heatmap
                ax_heatmap = fig.add_subplot(inner_grid[0])
                sns.heatmap(matrix,
                           annot=True,
                           fmt='.2f',
                           cmap=custom_cmap,
                           center=0.5,
                           vmin=0,
                           vmax=1,
                           square=True,
                           xticklabels=False,
                           yticklabels=solvers,
                           ax=ax_heatmap,
                           cbar_kws={'label': 'Comparison Result'},
                           cbar=False)
                
                # Plot barplot
                ax_barplot = fig.add_subplot(inner_grid[1])
                # Sort times for better visualization
                sorted_times = dict(sorted(times.items(), key=lambda x: x[1]))
                bars = ax_barplot.barh(list(range(len(sorted_times))), 
                                     list(sorted_times.values()),
                                     align='center')
                
                # Customize barplot
                ax_barplot.set_yticks(range(len(sorted_times)))
                ax_barplot.set_yticklabels(list(sorted_times.keys()))
                ax_barplot.set_xlabel('Time (s)')
                
                # Add value labels on bars
                for bar in bars:
                    width = bar.get_width()
                    ax_barplot.text(width, bar.get_y() + bar.get_height()/2,
                                  f'{width:.1f}',
                                  ha='left', va='center')
                
                # Set titles
                if i == 0:
                    ax_heatmap.set_title(f'{dataset}\nWilcoxon Matrix')
                    ax_barplot.set_title(f'{dataset}\nSolver Times')
                
                if j == 0:
                    ax_heatmap.set_ylabel(f'{classifier}')
                
                # Rotate labels
                ax_heatmap.tick_params(left=True, bottom=False)
                plt.setp(ax_heatmap.get_yticklabels(), rotation=0)
                
            else:
                # If no data, create empty subplot
                ax = fig.add_subplot(outer_grid[i, j])
                ax.axis('off')
    
    # Add overall title
    plt.suptitle('\n\n', fontsize=16, y=1.02)
    
    # Adjust layout
    # plt.tight_layout()
    plt.show()

def main(file_paths: List[str]):
    """Main function to process data and generate visualizations."""
    all_experiments = []
    
    # Read and process all experiment files
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
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
                    'accuracies': [solution['bestAccuracy'] for solution in experiment['solutions']],
                    'timeSec':exp_data['metadata']['elapsedTimeSec']
                })
            all_experiments.extend(experiments)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    # Create and process DataFrame
    df = pd.DataFrame(all_experiments)
    df = df.explode('accuracies').reset_index(drop=True)
    df['accuracies'] = df['accuracies'].astype(float)
    # groupedDF=group_experiments(df)

    # Generate plot
    plot_comparison_matrices(df)

if __name__ == "__main__":
    # Your existing file paths setup
    LOGS_ROOT = "/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs"
    problemCategories = ["classification"]
    models = ["RandomForest", "SVM", "GradientBoost", "DecisionTree"]
    # experiments = ["smallDatasets/smallIter"]
    # experiments=["smallDatasets/HybridBayes"]
    # experiments=["smallDatasets/biggerIter"]
    # solvers = ["pyNMHH", "bayesGP"]
    solverAndExperiment=[("defaultParameters","smallDatasets/defaultParams"),("gridSearch","smallDatasets/biggerIter"),("geneticSearch","smallDatasets/biggerIter"),("randomSearch","smallDatasets/smallIter"),("bayesGP","smallDatasets/smallIter"),("pyNMHH","smallDatasets/biggerIter")]
    paths=[]
    for problemCategory in problemCategories:
        for solver,experiment in  solverAndExperiment:
            for model in models:
                paths.append(f"{LOGS_ROOT}/{solver}/{problemCategory}/{model}/{experiment}/records.json" )
                # print(f"Category: {problemCategory}, model: {model}, experiment: {experiment}, solvers: {solvers}")
    main(paths)