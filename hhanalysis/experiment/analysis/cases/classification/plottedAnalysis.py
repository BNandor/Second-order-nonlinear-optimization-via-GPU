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
from collections import Counter

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
                
                if max1-max2>0.0025:
                    matrix[i, j] = 1.0 
                elif abs(max1-max2)<0.0025:
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

def capitalize_first_letter(text):
    return text[:1].upper() + text[1:] if text else text

def plot_comparison_matrices(df, max_datasets_per_figure=2,pyNMHHSolvers={},proposedSolver='pyNMHH'):
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
                pyNMHHSubset=subset[(subset['solver']==proposedSolver)]
                
                if not subset.empty:
                    # Create inner gridspec for matrix and barplot
                    inner_grid = outer_grid[i, j].subgridspec(1, 3, width_ratios=[1, 1.0,1.0])
                    
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
                               yticklabels=list(map(lambda s:capitalize_first_letter(s.replace('defaultParameters','default')).replace(proposedSolver,'NMHT'),solvers)),
                               ax=ax_heatmap,
                               cbar_kws={'label': 'Comparison Result'},
                               cbar=False)
                    ax_heatmap.set_xlabel('Wins')

                    # Plot barplot
                    ax_barplot = fig.add_subplot(inner_grid[1])
                    # Sort times for better visualization
                    sorted_times = dict(sorted(times.items(), key=lambda x: x[0],reverse=True))
                    # colors = cm.coolwarm(np.linspace(1, 0, len(solvers)))
                    bars = ax_barplot.barh(list(range(len(sorted_times))), 
                                         list(sorted_times.values()),
                                         align='center')
                    ax_barplot.set_xlim(0, max(sorted_times.values()) * 1.7)
                    
                    # Customize barplot
                    ax_barplot.set_yticks(range(len(sorted_times)))
                    # ax_barplot.set_yticklabels(list(map(lambda s:capitalize_first_letter(s.replace('defaultParameters','default')),sorted_times.keys())))
                    ax_barplot.set_yticklabels( [ "" for k in sorted_times.keys()])
                    # ax_barplot.tick_params(axis='both', labelsize=7)
                    ax_barplot.set_xlabel('Time (s)')
                    
                    # Add value labels on bars
                    for bar in bars:
                        width = bar.get_width()
                        ax_barplot.text(width, bar.get_y() + bar.get_height()/2,
                                      f'{width:.1f}',
                                      ha='left', va='center')
                    pos = ax_barplot.get_position()
                    ax_barplot.set_position([pos.x0 -0.015, pos.y0, pos.width, pos.height])
                    # ax_barplot.set_position([pos.x0-1000, pos.y0, pos.width, pos.height])
                    # Operator statistics   
                    ax_barplot_operators = fig.add_subplot(inner_grid[2])
                    
                    operatorsUsed= pyNMHHSubset['solverOperators'].values[0] if len(pyNMHHSubset['solverOperators'].values)>0 else {}
                    for solver in pyNMHHSolvers:
                        if solver not in operatorsUsed:
                            operatorsUsed[solver]=0

                    sorted_operators = dict(sorted(operatorsUsed.items(),reverse=True))
                    # colors = cm.coolwarm(np.linspace(1, 0, len(solvers)))
                    operatorBars = ax_barplot_operators.barh(list(range(len(sorted_operators))), 
                                         list(sorted_operators.values()),color=cm.viridis(np.linspace(0, 1, len(sorted_operators))),
                                         align='center')
                    ax_barplot_operators.set_xlim(0, max(sorted_operators.values())+max(sorted_operators.values())/3)
                    
                    # Customize barplot
                    ax_barplot_operators.set_yticks(range(len(sorted_operators)))
                    ax_barplot_operators.set_yticklabels(list(map(lambda s:capitalize_first_letter(s).replace('Best','Elitist'),sorted_operators.keys())))
                    ax_barplot_operators.set_xlabel('Frequency')
                    pos = ax_barplot_operators.get_position()
                    ax_barplot_operators.set_position([pos.x0 +0.01, pos.y0, pos.width, pos.height])
                    # Set titles
                    if i == 0:
                        ax_heatmap.set_title(f'{dataset}\n Results')
                        ax_barplot.set_title(f'{dataset}\nSolver Times')
                        ax_barplot_operators.set_title('pyNMHH\n Operator Frequencies')
                    
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
        # plt.tight_layout()
        # plt.subplots_adjust(
        #     left=0.1,    # left margin
        #     right=0.9,   # right margin
        #     bottom=0.1,  # bottom margin
        #     top=0.9,     # top margin
        #     wspace=0.5,  # width spacing between subplots
        #     hspace=0.4   # height spacing between subplots
        # )

        # Show the figure (or save if preferred)
        plt.show()

def getBestSolutionStatistics(experiment,pyNMHHSolvers,proposedSolver):
    if experiment['solver'] != proposedSolver:
        return {}
    bestSolution=max(experiment['solutions'],key=lambda s:s['bestAccuracy'])
    bestSequence=bestSolution['bestSequence']
    frequencies = Counter(bestSequence)
    dfrequencies=dict(frequencies)
    pyNMHHSolvers.update(list(dfrequencies.keys()))
    return dfrequencies

def printExperimentTable(df):
    experiments=df[["datasetName","classifierModel","solver","accuracies"]]
    result = experiments.groupby(["datasetName",'classifierModel', 'solver'])['accuracies'].max().reset_index()
    # Convert to LaTeX format with some styling
    pivot_table = result.pivot(
        index=['datasetName','classifierModel'],
        columns='solver',
        values='accuracies'
    ).reset_index()
    # pivot_table=pivot_table[['datasetName','classifierModel','pyNMHH','bayesGP','bayesTPE', 'defaultParameters','geneticSearch','gridSearch','randomSearch']]
    latex_table = pivot_table.to_latex(
        index=False,
        float_format="%.5f",
        caption="Maximum Accuracies by Classifier and Dataset",
        label="tab:accuracies_by_solver",
        escape=True
    )

    # Improve formatting
    latex_table = latex_table.replace('tabular', 'tabular*{\\textwidth}')
    latex_table = latex_table.replace('classifierModel', 'Classifier')
    latex_table = latex_table.replace('datasetName', 'Dataset')
    print(latex_table)

def main(file_paths: List[str],datasets_filter=[],proposedSolver='pyNMHH'):
    """Main function to process data and generate visualizations."""
    all_experiments = []
    pyNMHHSolvers=set()

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
                    'timeSec':exp_data['metadata']['elapsedTimeSec'],
                    'solverOperators':getBestSolutionStatistics(experiment,pyNMHHSolvers,proposedSolver)
                })
            all_experiments.extend(experiments)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    # Create and process DataFrame
    df = pd.DataFrame(all_experiments)
    if datasets_filter:
        df = df[df["datasetName"].isin(datasets_filter)]
    
    df = df.explode('accuracies').reset_index(drop=True)

    df['accuracies'] = df['accuracies'].astype(float)
    # groupedDF=group_experiments(df)

    # Generate plot
    printExperimentTable(df)
    plotWins(df)
    plot_comparison_matrices(df,2,pyNMHHSolvers,proposedSolver)

if __name__ == "__main__":
    # Your existing file paths setup
    LOGS_ROOT = "../../../../logs"
    problemCategories = ["classification"]
    models = [
              "RandomForest", 
              "GradientBoost", 
              "DecisionTree",
              "SVM"
            ]
    datasets=[ 
        'Digits',
        'Wine',
        'AuditRisk',
        'CervicalCancer',
        'GallStone',
        'HigherEducation'
        ]
    proposedSolver='pyNMHH_HALF_SA'
    solverAndExperiment=[
                         ("defaultParameters","smallDatasets/defaultParams"),
                         ("gridSearch","smallDatasets/biggerIter"),
                         ("geneticSearch","smallDatasets/biggerIter"),
                         ("randomSearch","smallDatasets/smallIter"),
                         ("bayesGP","smallDatasets/pyNMHHBased"),
                         ("bayesTPE","smallDatasets/biggerIter"),

                        #  ("pyNMHH","smallDatasets/biggerIter/smallerPop"),
                        # ("pyNMHH","smallDatasets/biggerIter/smallerPop/bayesinit"),
                        
                        # Don't forget to update contents if experiments changed
                        # ("pyNMHH","smallDatasets/biggerIter/smallerPop/_combined"),
                        ('pyNMHH_HALF_SA','smallDatasets/halfSA/smallerPop/bayesinit')
                         ]
    # solverAndExperiment=[("bayesGP","smallDatasets/pyNMHHBased"),("pyNMHH","smallDatasets/biggerIter/smallerPop")]
    paths=[]
    for problemCategory in problemCategories:
        for solver,experiment in  solverAndExperiment:
            for model in models:
                paths.append(f"{LOGS_ROOT}/{solver}/{problemCategory}/{model}/{experiment}/records.json" )
                # print(f"Category: {problemCategory}, model: {model}, experiment: {experiment}, solvers: {solvers}")
    main(paths,datasets,proposedSolver)