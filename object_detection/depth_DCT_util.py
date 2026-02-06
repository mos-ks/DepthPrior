import os

import numpy as np
import pandas as pd


def parse_latex_table(file_path):
    """
    Parse a LaTeX table and return a pandas DataFrame.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract rows between \midrule and \bottomrule
    data_lines = []
    start, end = False, False
    for line in lines:
        if "\\midrule" in line:
            start = True
            continue
        if "\\bottomrule" in line:
            end = True
        if start and not end:
            data_lines.append(line.strip()[:-2])
    
    # Parse rows into a DataFrame
    data = []
    for row in data_lines:
        if row:
            data.append([float(x) if '.' in x or x.isdigit() else int(x) for x in row.split('&')])
    
    columns = ["tau_0", "static_TD", "static_ED", "optimized_TD", "optimized_ED", "Delta_TD", "Delta_ED"]
    df = pd.DataFrame(data, columns=columns)
    return df


def process_files(file_paths):
    """
    Process multiple LaTeX table files and calculate totals, means, and std deviations.
    """
    summary_rows = []
    all_totals = []

    for file_path in file_paths:
        df = parse_latex_table(file_path)
        totals = df.sum(axis=0)
        totals["tau_0"] = "/".join(file_path.split("plots_DCT")[1].split("/")[1:3])
        summary_rows.append(totals)        
        all_totals.append(np.array(totals[1:], dtype=float))  # Ensure numeric type


    # Create a DataFrame for summary rows
    summary_df = pd.DataFrame(summary_rows)

    # Calculate overall totals, means, and std deviations
    overall_totals = pd.Series(np.sum(all_totals, axis=0), index=summary_df.columns[1:])
    overall_totals["tau_0"] = "Total"

    # Combine all rows into a final DataFrame
    final_df = pd.concat([summary_df, overall_totals.to_frame().T], ignore_index=True)
    return final_df


def generate_latex_table(df, output_path):
    """
    Generate a LaTeX table from the DataFrame and save it to a file.
    """
    with open(output_path, 'w') as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Setup & static TD & static ED & optimized TD & optimized ED & $\\Delta_{TD}$ & $\\Delta_{ED}$ \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['tau_0']} & {int(row['static_TD'])} & {int(row['static_ED'])} & {int(row['optimized_TD'])} & {int(row['optimized_ED'])} & {int(row['Delta_TD'])} & {int(row['Delta_ED'])} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    # Define the directory containing LaTeX table files
    directory = "/path/to/data/DepthPrior/object_detection/plots_DCT/"
    file_name = 'spline_results_table.tex' #performance_table for val, spline_results_table for infer, 
    file_paths = [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if f.endswith(file_name)]

    # Process files and generate the final summary DataFrame
    final_df = process_files(file_paths)

    # Save the final DataFrame as a LaTeX table
    output_path = os.path.join(directory, f"summary_{file_name}")
    generate_latex_table(final_df, output_path)
    print(f"LaTeX table saved to {output_path}")


if __name__ == "__main__":
    main()