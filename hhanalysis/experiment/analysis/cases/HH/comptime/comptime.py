"""
Computational Time Analysis Module

Aggregates elapsed times from optimization experiments across multiple records.json files.
Computes mean and standard deviation of execution times grouped by problem and dimension.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_json(path):
    """Load JSON file from path."""
    with open(path, 'r') as f:
        return json.load(f)


def discover_records_files(root_path):
    """
    Recursively discover all records.json files under root_path.
    
    Args:
        root_path (str): Root directory to search
        
    Returns:
        list: List of absolute paths to records.json files
    """
    records_files = []
    for root, dirs, files in os.walk(root_path):
        if 'records.json' in files:
            records_files.append(os.path.join(root, 'records.json'))
    return sorted(records_files)


def extract_experiment_data(experiment_dict):
    """
    Extract relevant fields from a single experiment record.
    
    Args:
        experiment_dict (dict): Single experiment from records.json
        
    Returns:
        dict: Extracted data with keys: problem, dimension, elapsed_time
    """
    experiment_config = experiment_dict.get('experiment', {})
    metadata = experiment_dict.get('metadata', {})
    
    # Extract problem name and strip 'PROBLEM_' prefix
    problem_name = experiment_config.get('problems', [None])[0]
    if problem_name and problem_name.startswith('PROBLEM_'):
        problem_name = problem_name[8:]  # Remove 'PROBLEM_' prefix
    
    # Normalize problem name to uppercase for consistency
    if problem_name:
        problem_name = problem_name.upper()
    
    # Extract dimension (modelSize)
    dimension = experiment_config.get('modelSize', None)
    
    # Extract elapsed time
    elapsed_time = metadata.get('elapsedTimeSec', None)
    
    return {
        'problem': problem_name,
        'dimension': dimension,
        'elapsed_time': elapsed_time
    }


def aggregate_elapsed_times(root_path):
    """
    Recursively discover all records.json files and aggregate elapsed times
    by problem and dimension.
    
    Args:
        root_path (str): Root directory path containing records.json files
        
    Returns:
        pd.DataFrame: Aggregated data with columns:
                      - problem: Problem name (without PROBLEM_ prefix)
                      - dimension: Problem dimension
                      - mean_time: Mean elapsed time in seconds
                      - std_time: Standard deviation of elapsed time
                      - count: Number of measurements
    """
    # Discover all records.json files
    records_files = discover_records_files(root_path)
    
    if not records_files:
        raise FileNotFoundError(f"No records.json files found under {root_path}")
    
    print(f"Found {len(records_files)} records.json files")
    
    # Aggregate all experiment data
    all_data = []
    
    for records_file in records_files:
        try:
            data = load_json(records_file)
            experiments = data.get('experiments', {})
            
            for exp_id, exp_data in experiments.items():
                extracted = extract_experiment_data(exp_data)
                all_data.append(extracted)
        except Exception as e:
            print(f"Warning: Error reading {records_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid experiment data found in records.json files")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Group by problem and dimension, compute statistics
    grouped = df.groupby(['problem', 'dimension'])['elapsed_time'].agg([
        ('mean_time', 'mean'),
        ('std_time', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    # Sort by problem name and dimension for better readability
    grouped = grouped.sort_values(['problem', 'dimension']).reset_index(drop=True)
    
    return grouped


def aggregate_multiple_methods(method_configs):
    """
    Aggregate computational times for multiple methods and combine into a single DataFrame.
    
    Args:
        method_configs (list): List of tuples (method_name, path) where:
                              - method_name (str): Name of the method (e.g., "SA-NMHH", "CustomHyS")
                              - path (str): Path containing records.json files
        
    Returns:
        pd.DataFrame: Combined data with columns:
                      - problem: Problem name
                      - method_{method_name}_mean: Mean time for each method
                      - method_{method_name}_std: Std time for each method
    """
    method_dfs = {}
    
    # Process each method
    for method_name, path in method_configs:
        print(f"\nProcessing method: {method_name}")
        print(f"Path: {path}")
        
        try:
            df = aggregate_elapsed_times(path)
            # Create columns for this method
            df_method = df[['problem', 'dimension', 'mean_time', 'std_time', 'count']].copy()
            df_method.columns = ['problem', 'dimension', f'{method_name}_mean', f'{method_name}_std', f'{method_name}_count']
            method_dfs[method_name] = df_method
        except Exception as e:
            print(f"Error processing {method_name}: {e}")
            continue
    
    if not method_dfs:
        raise ValueError("No methods were successfully processed")
    
    # Merge all method DataFrames on problem and dimension
    combined_df = None
    for method_name, df_method in method_dfs.items():
        if combined_df is None:
            combined_df = df_method
        else:
            combined_df = combined_df.merge(
                df_method,
                on=['problem', 'dimension'],
                how='outer'
            )
    
    # Sort by problem and dimension
    combined_df = combined_df.sort_values(['problem', 'dimension']).reset_index(drop=True)
    
    return combined_df


def aggregate_by_problem_only(combined_df, method_names):
    """
    Aggregate computational times across dimensions for each method and problem.
    Creates a summary with one row per problem showing mean/std across all dimensions.
    
    Args:
        combined_df (pd.DataFrame): Combined DataFrame from aggregate_multiple_methods()
        method_names (list): List of method names (for column identification)
        
    Returns:
        pd.DataFrame: Aggregated data with columns:
                      - problem: Problem name
                      - method_{method_name}_mean: Mean time across dimensions
                      - method_{method_name}_std: Std time across dimensions
    """
    # Group by problem only, computing mean of means for each method
    result_rows = []
    
    for problem in combined_df['problem'].unique():
        problem_data = combined_df[combined_df['problem'] == problem]
        row = {'problem': problem}
        
        for method_name in method_names:
            mean_col = f'{method_name}_mean'
            std_col = f'{method_name}_std'
            
            if mean_col in problem_data.columns:
                # Average of means across dimensions
                row[f'{method_name}_mean'] = problem_data[mean_col].mean() 
                # Compute combined std across dimensions
                row[f'{method_name}_std'] = problem_data[std_col].mean()
        
        result_rows.append(row)
    
    summary_df = pd.DataFrame(result_rows)
    summary_df = summary_df.sort_values('problem').reset_index(drop=True)
    
    return summary_df


def to_latex_table(df, caption="Computational Time Analysis", label="tab:comptime"):
    """
    Convert aggregated data to LaTeX table format.
    
    Args:
        df (pd.DataFrame): Aggregated data from aggregate_elapsed_times()
        caption (str): LaTeX table caption
        label (str): LaTeX table label
        
    Returns:
        str: LaTeX table code
    """
    # Format the data for better readability
    df_latex = df.copy()
    df_latex['mean_time'] = df_latex['mean_time'].apply(lambda x: f"{x:.2f}")
    df_latex['std_time'] = df_latex['std_time'].apply(lambda x: f"{x:.2f}")
    df_latex['count'] = df_latex['count'].astype(int)
    
    # Rename columns for LaTeX
    df_latex.columns = ['Problem', 'Dimension', 'Mean (s)', 'Std (s)', 'N']
    
    # Generate LaTeX table
    latex_str = df_latex.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=False,
        float_format=lambda x: f"{x:.2f}"
    )
    
    return latex_str


def format_mean_std(mean_val, std_val, bold=False):
    """Format mean and std as 'mean $\\pm$ std' (or mean only if std is NaN), optionally bolded."""
    if pd.notna(mean_val):
        if pd.notna(std_val):
            formatted = f"{mean_val:.2f} $\\pm$ {std_val:.2f}"
        else:
            formatted = f"{mean_val:.2f}"
        if bold:
            formatted = f"\\textbf{{{formatted}}}"
        return formatted
    return "---"


def to_latex_comparison_table(df, method_names, by_problem=False, caption="Computational Time Comparison", label="tab:comptime_comparison"):
    """
    Convert comparison data to LaTeX table format with mean±std notation and multirow for problems.
    
    Args:
        df (pd.DataFrame): Combined DataFrame from aggregate_multiple_methods() or aggregate_by_problem_only()
        method_names (list): List of method names for formatting
        by_problem (bool): If True, only show problem column; if False, show problem and dimension
        caption (str): LaTeX table caption
        label (str): LaTeX table label
        
    Returns:
        str: LaTeX table code with multirow for problems
    """
    df_latex = df.copy()
    
    # Convert mean±std format for each method
    for method_name in method_names:
        mean_col = f'{method_name}_mean'
        std_col = f'{method_name}_std'
        
        if mean_col in df_latex.columns and std_col in df_latex.columns:
            df_latex[f'{method_name}'] = df_latex.apply(
                lambda row: format_mean_std(row[mean_col], row[std_col]),
                axis=1
            )
            # Drop the separate mean and std columns
            df_latex = df_latex.drop(columns=[mean_col, std_col])
    
    # Handle by_problem case (no multirow needed)
    if by_problem:
            # Bold the minimum value in each row
        for idx, row in df_latex.iterrows():
                # Find minimum mean across methods for this row
                method_values = []
                for method_name in method_names:
                    mean_col = f'{method_name}_mean'
                    if mean_col in df.columns:
                        method_values.append((method_name, df.loc[idx, mean_col]))
            
                # Find which method has minimum mean
                valid_means = [(m, v) for m, v in method_values if pd.notna(v)]
                min_method = min(valid_means, key=lambda x: x[1])[0] if valid_means else None
            
                # Bold the minimum in the formatted row
                for method_name in method_names:
                    if method_name == min_method:
                        current_val = df_latex.loc[idx, method_name]
                        if isinstance(current_val, str) and '---' not in current_val:
                            df_latex.loc[idx, method_name] = f"\\textbf{{{current_val}}}"

        df_latex.columns = ['Problem' if c == 'problem' else c for c in df_latex.columns]
        latex_str = df_latex.to_latex(
            index=False,
            caption=caption,
            label=label,
            escape=False
        )
        return latex_str
    
    # For detailed comparison with multirow
    # Group by problem and create multirow formatting
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    
    # Build column specification
    num_methods = len(method_names)
    col_spec = "|l|r|" + "|".join(["c" for _ in range(num_methods)]) + "|"
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\hline")
    
    # Build header
    header = "Problem & Dimension & " + " & ".join(method_names) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")
    
    # Group by problem and write rows with multirow
    problems = df_latex['problem'].unique()
    for problem in problems:
        problem_indices = df_latex[df_latex['problem'] == problem].index
        problem_data = df_latex.loc[problem_indices].reset_index(drop=True)
        num_rows = len(problem_data)
        
        for idx, (_, row) in enumerate(problem_data.iterrows()):
                # Find minimum mean across methods for this row (use original df)
                original_idx = problem_indices[idx]
                mean_values = []
                for method_name in method_names:
                    mean_col = f'{method_name}_mean'
                    if mean_col in df.columns:
                        mean_values.append((method_name, df.loc[original_idx, mean_col]))
            
                # Find which method has minimum mean (only consider valid numbers)
                valid_means = [(m, v) for m, v in mean_values if pd.notna(v)]
                min_method = min(valid_means, key=lambda x: x[1])[0] if valid_means else None

                # Add multirow for problem on first row only
                if idx == 0 and num_rows > 1:
                    problem_str = f"\\multirow{{{num_rows}}}{{*}}{{{row['problem']}}}"
                elif idx == 0:
                    problem_str = row['problem']
                else:
                    problem_str = ""
                
                # Build row data
                dimension = row['dimension']
                method_values_list = []
                for method_name in method_names:
                    value_str = str(row[method_name])
                    if method_name == min_method:
                        value_str = f"\\textbf{{{value_str}}}"
                    method_values_list.append(value_str)
                
                method_values = " & ".join(method_values_list)
                
                if problem_str:
                    latex_lines.append(f"{problem_str} & {dimension} & {method_values} \\\\")
                else:
                    latex_lines.append(f" & {dimension} & {method_values} \\\\")
            
        latex_lines.append("\\hline")
    
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def save_results(df, output_dir, base_name="comptime_aggregated"):
    """
    Save aggregation results to CSV and LaTeX formats.
    
    Args:
        df (pd.DataFrame): Aggregated data
        output_dir (str): Directory to save output files
        base_name (str): Base name for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, f"{base_name}.tex")
    latex_str = to_latex_table(df)
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved LaTeX to {latex_path}")
    
    return csv_path, latex_path

def compare(method_configs):
    print("=" * 80)
    print("COMPUTATIONAL TIME COMPARISON")
    print("=" * 80)
    
    # Aggregate multiple methods
    combined_df = aggregate_multiple_methods(method_configs)
    method_names = [name for name, _ in method_configs]
    # Calculate average mean time for each method
    print("\n" + "=" * 80)
    print("Average Mean Time Across All Problems and Dimensions:")
    print("=" * 80)
    for method_name in method_names:
            mean_col = f'{method_name}_mean'
            if mean_col in combined_df.columns:
                avg_mean = combined_df[mean_col].mean()
                print(f"{method_name}: {avg_mean:.2f} seconds")
    print("\n" + "=" * 80)
    print("Combined Results (by Problem and Dimension):")
    print("=" * 80)
    print(combined_df)
    
    # Generate LaTeX table for detailed results
    print("\n" + "=" * 80)
    print("LaTeX Table (Detailed - by Problem and Dimension):")
    print("=" * 80)
    latex_detailed = to_latex_comparison_table(
        combined_df, 
        method_names,
        by_problem=False,
        caption="Computational Time Comparison by Problem and Dimension",
        label="tab:comptime_detailed"
    )
    print(latex_detailed)
    
    # Aggregate by problem only (average across dimensions)
    print("\n" + "=" * 80)
    print("Aggregated by Problem (mean across dimensions):")
    print("=" * 80)
    summary_df = aggregate_by_problem_only(combined_df, method_names)
    print(summary_df)
    
    # Generate LaTeX table for summary
    print("\n" + "=" * 80)
    print("LaTeX Table (Summary - by Problem Only):")
    print("=" * 80)
    latex_summary = to_latex_comparison_table(
        summary_df,
        method_names,
        by_problem=True,
        caption="Computational Time Comparison by Problem (averaged across dimensions)",
        label="tab:comptime_summary"
    )
    print(latex_summary)

# Example usage
if __name__ == "__main__":
    # NMHH comparison
    # method_configs = [
    #     ("SA-NMHH", "/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs/SA-NMHH/GA_DE_GD_LBFGS/comptime"),
    #     ("CustomHyS", "/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs/CustomHYSPerf/newExperiment/comptime"),
    # ]
    # NMHH, NMHH-SPRT comparison 10 hyper steps
    method_configs = [
        ("SA-NMHH-SPRT", "/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs/SA-NMHH/GA_DE_GD_LBFGS/sprt/comptime_extended_aws"),
        ("SA-NMHH", "/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs/SA-NMHH/GA_DE_GD_LBFGS/comptime_extended_aws"),
        # ("CustomHyS", "/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs/CustomHYSPerf/newExperiment/comptime"),
    ]
    # # NMHH Hybrid methods comparison
    # method_configs = [
    #     ("SA-NMHH", "/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs/SA-NMHH/GA_DE_GD_LBFGS/comptime_extended_initial"),
    #     ("mealpy",  "/home/spaceman/dissertation/finmat/ParallelLBFGS/hhanalysis/logs/SA-NMHH/GA_DE_GD_LBFGS/comptime_extended_initial"),
        
    # ]
    compare(method_configs)
   
    
    # Optional: Save results
    # output_dir = "/path/to/output"
    # os.makedirs(output_dir, exist_ok=True)
    # combined_df.to_csv(os.path.join(output_dir, "comptime_detailed.csv"), index=False)
    # summary_df.to_csv(os.path.join(output_dir, "comptime_summary.csv"), index=False)
    # with open(os.path.join(output_dir, "comptime_detailed.tex"), 'w') as f:
    #     f.write(latex_detailed)
    # with open(os.path.join(output_dir, "comptime_summary.tex"), 'w') as f:
    #     f.write(latex_summary)

