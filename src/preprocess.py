import pandas as pd
import numpy as np
import glob
import os

def main():
    print("--- STARTING DATA PREPROCESSING ---")
    data_dir = r"d:/Projects/bridge-optimization/data"
    output_path = os.path.join(data_dir, "preprocessed_optimal_designs.csv")
    
    # 1. Find and load all CSV files
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    # Exclude output files if they exist
    csv_files = [f for f in csv_files if os.path.basename(f) not in ["preprocessed_optimal_designs.csv", "optimal_designs.csv"]]
    
    if not csv_files:
        print("Error: No raw CSV files found in the data directory!")
        return
        
    dfs = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Loading {filename}...")
        df_temp = pd.read_csv(file_path)
        dfs.append(df_temp)
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Merged Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Standardize categorical features & handle NaNs
    df['footpath'] = df['footpath'].fillna('None').astype(str).str.strip()
    df['steel_grade'] = df['steel_grade'].astype(str).str.strip()
    df['concrete_grade'] = df['concrete_grade'].astype(str).str.strip()
    
    # 3. Filter for feasible designs (PASS only)
    pass_df = df[df['status'] == 'PASS'].copy().reset_index(drop=True)
    print(f"Feasible Designs: {pass_df.shape[0]} rows")
    
    # 4. Group by all 6 inputs and select the minimum weight design
    group_cols = ['span_m', 'carriageway_width_m', 'footpath', 'skew_angle_deg', 'steel_grade', 'concrete_grade']
    best_idx = pass_df.groupby(group_cols)['total_kN'].idxmin()
    best_designs = pass_df.loc[best_idx].copy().reset_index(drop=True)
    print(f"Extracted optimal designs for {best_designs.shape[0]} unique configurations")
    
    # 5. Physics-based categorical encoding
    footpath_map = {'None': 0, 'Single Side': 1, 'Both Sides': 2}
    steel_map = {'E 350A': 350, 'E 410A': 410, 'E 450A': 450}
    concrete_map = {'M40': 40, 'M50': 50, 'M60': 60}
    
    best_designs['footpath_encoded'] = best_designs['footpath'].map(footpath_map)
    best_designs['steel_grade_encoded'] = best_designs['steel_grade'].map(steel_map)
    best_designs['concrete_grade_encoded'] = best_designs['concrete_grade'].map(concrete_map)
    
    # Verify mappings succeeded
    assert best_designs['footpath_encoded'].isnull().sum() == 0, "Error mapping footpath!"
    assert best_designs['steel_grade_encoded'].isnull().sum() == 0, "Error mapping steel_grade!"
    assert best_designs['concrete_grade_encoded'].isnull().sum() == 0, "Error mapping concrete_grade!"
    
    # 6. Monotonicity Filtering (Skipped per user/mentor feedback)
    print("\nMonotonicity check skipped: Shorter bridges are allowed to be heavier.")
    clean_designs = best_designs.copy().reset_index(drop=True)
    
    # 7. Save output
    # Select columns to keep in order
    output_cols = [
        'span_m', 'carriageway_width_m', 'footpath', 'footpath_encoded', 
        'skew_angle_deg', 'steel_grade', 'steel_grade_encoded', 
        'concrete_grade', 'concrete_grade_encoded',
        'n', 'spacing_m', 't_slab_mm', 'D_mm', 'bf_mm', 'tf_mm', 'tw_mm', 'dw_mm',
        'steel_kN', 'deck_kN', 'total_kN'
    ]
    clean_designs = clean_designs[output_cols]
    clean_designs.to_csv(output_path, index=False)
    print(f"Preprocessed data successfully saved to: {output_path}")
    print(f"Processed dataset shape: {clean_designs.shape[0]} rows, {clean_designs.shape[1]} columns")
    
    # Print sample
    print("\n--- SAMPLE PREPROCESSED DATA (FIRST 5 ROWS) ---")
    print(clean_designs.head().to_string(index=False))

if __name__ == '__main__':
    main()
