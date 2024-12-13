import json
import os
from typing import List, Dict, Any
import hashlib
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
                
                if max1-max2>0.001:
                    matrix[i, j] = 1.0 
                elif abs(max1-max2)<0.001:
                    # 0.5 means no significant difference
                    matrix[i, j] = 0.5
                else: 
                    matrix[i, j] = 0.0

    return matrix, solvers

def getSolverTimes(group,solvers):
    allSolverTimes={}                
    for i, solver in enumerate(solvers):
        solverTimes=group[group['solver'] == solver]['timeSec'].array
        allSolverTimes[solver]=np.sum(solverTimes) #solverTimes[0]/len(solverTimes)
    return allSolverTimes

def calculate_solver_wins(df):
    """Calculate total wins for each solver across all experiments."""
    solver_wins = {}
    
    # Group by classifier model and dataset
    grouped = df.groupby(['classifierModel', 'datasetName'])
    
    for (classifier, dataset), group in grouped:
        # Create matrix and get solvers
        matrix, solvers = create_maxcomparison_matrix(group)
        
        # Count wins for each solver
        for i, solver in enumerate(solvers):
            wins = np.sum(matrix[i, :] == 1.0)
            solver_wins.setdefault(solver, 0)
            solver_wins[solver] += wins
    
    return solver_wins

def plotWins(df):
    solver_wins = calculate_solver_wins(df)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort solvers by number of wins in descending order
    sorted_solver_wins = dict(sorted(solver_wins.items(), key=lambda x: x[1], reverse=True))
    solvers = list(sorted_solver_wins.keys())
    wins = list(sorted_solver_wins.values())

    colors = cm.coolwarm(np.linspace(0, 1, len(solvers)))

    # Create horizontal bar plot with different colors
    bars = ax.barh(solvers, wins, color=colors)


    # Customize the plot
    ax.set_xlabel('Score')
    ax.set_title('Solver Performance')

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', 
                ha='left', va='center')

    ax.set_xlim(0, max(wins) * 1.1)

    plt.title('Solver scores')
    plt.ylabel('Solver')
    plt.xlabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return solver_wins

def group_experiments(df: pd.DataFrame):
    """Group experiments based on specified fields."""
    # return df.groupby(['totalFunctionEvaluations', 'classifierModel', 
    #                    'datasetName', 'hyperParameters_hash'])
    return df.groupby(['classifierModel','datasetName','hyperParameters_hash'])

def plot_comparison_matrices(df, max_datasets_per_figure=2):
    """
    Plot comparison matrices with time bar plots, splitting into multiple figures 
    if there are more datasets than max_datasets_per_figure.
    """
    # Get unique values for grid organization
    classifiers = sorted(df['classifierModel'].unique())
    datasets = sorted(df['datasetName'].unique())
    
    # Calculate number of figures needed
    num_figures = (len(datasets) + max_datasets_per_figure - 1) // max_datasets_per_figure
    
    # Create custom black and white colormap
    custom_cmap = plt.cm.get_cmap('gray_r')  # reverse gray colormap
    
    # Iterate through figures
    for fig_idx in range(num_figures):
        # Select datasets for this figure
        start_idx = fig_idx * max_datasets_per_figure
        end_idx = min(start_idx + max_datasets_per_figure, len(datasets))
        current_datasets = datasets[start_idx:end_idx]
        
        # Create figure
        fig = plt.figure(figsize=(8 * len(current_datasets), 5 * len(classifiers)))
        
        # Create a grid with proper spacing
        outer_grid = fig.add_gridspec(len(classifiers), len(current_datasets), 
                                      hspace=0.3, wspace=0.4)
        
        # Plot each subplot
        for i, classifier in enumerate(classifiers):
            for j, dataset in enumerate(current_datasets):
                # Filter data for this classifier-dataset combination
                subset = df[(df['classifierModel'] == classifier) &
                           (df['datasetName'] == dataset)]
                
                if not subset.empty:
                    # Create inner gridspec for matrix and barplot
                    inner_grid = outer_grid[i, j].subgridspec(1, 2, width_ratios=[1.5, 1])
                    
                    # Create matrix and get solver names and times
                    matrix, solvers = create_maxcomparison_matrix(subset)
                    times = getSolverTimes(subset,solvers)
                    
                    # Plot heatmap
                    ax_heatmap = fig.add_subplot(inner_grid[0])
                    sns.heatmap(matrix,
                               annot=False,  # Removed cell numbers
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
                    sorted_times = dict(sorted(times.items(), key=lambda x: x[0],reverse=True))
                    # colors = cm.coolwarm(np.linspace(1, 0, len(solvers)))
                    bars = ax_barplot.barh(list(range(len(sorted_times))), 
                                         list(sorted_times.values()),
                                         align='center')
                    ax_barplot.set_xlim(0, max(sorted_times.values()) * 1.4)
                    
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
                        ax_heatmap.set_title(f'{dataset}\n Results')
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
        
        # Add figure title to indicate which datasets are included
        plt.suptitle(f'Datasets: {", ".join(current_datasets)}', fontsize=16)
        
        # Show the figure (or save if preferred)
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
    plotWins(df)
    plot_comparison_matrices(df)

if __name__ == "__main__":
    # Your existing file paths setup
    LOGS_ROOT = "../../../../logs"
    problemCategories = ["classification"]
    models = ["RandomForest", "SVM", "GradientBoost", "DecisionTree"]
    # experiments = ["smallDatasets/smallIter"]
    # experiments=["smallDatasets/HybridBayes"]
    # experiments=["smallDatasets/biggerIter"]
    # solvers = ["pyNMHH", "bayesGP"]
    solverAndExperiment=[("defaultParameters","smallDatasets/defaultParams"),("gridSearch","smallDatasets/biggerIter"),("geneticSearch","smallDatasets/biggerIter"),("randomSearch","smallDatasets/smallIter"),("bayesGP","smallDatasets/bigIterEstimatorFix"),("bayesTPE","smallDatasets/biggerIter"),("pyNMHH","smallDatasets/biggerIter")]
    solverAndExperiment=[("defaultParameters","smallDatasets/defaultParams"),("gridSearch","smallDatasets/biggerIter"),("geneticSearch","smallDatasets/biggerIter"),("randomSearch","smallDatasets/smallIter"),("bayesGP","smallDatasets/pyNMHHBased"),("bayesTPE","smallDatasets/biggerIter"),("pyNMHH","smallDatasets/biggerIter/smallerPop")]
    paths=[]
    for problemCategory in problemCategories:
        for solver,experiment in  solverAndExperiment:
            for model in models:
                paths.append(f"{LOGS_ROOT}/{solver}/{problemCategory}/{model}/{experiment}/records.json" )
                # print(f"Category: {problemCategory}, model: {model}, experiment: {experiment}, solvers: {solvers}")
    main(paths)