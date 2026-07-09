import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    print("--- STARTING MULTI-VARIABLE MODEL TRAINING WORKFLOW ---")
    data_path = r"d:/Projects/bridge-optimization/data/preprocessed_optimal_designs.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Preprocessed dataset file not found at {data_path}")
        return
        
    # 1. Load Data
    df = pd.read_csv(data_path)
    print(f"Loaded preprocessed dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Define features and targets
    feature_cols = [
        'span_m', 'carriageway_width_m', 'footpath_encoded', 
        'skew_angle_deg', 'steel_grade_encoded', 'concrete_grade_encoded'
    ]
    target_cols = ['n', 'spacing_m', 't_slab_mm', 'D_mm', 'bf_mm', 'tf_mm', 'tw_mm']
    
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    
    # Calculate min and max bounds for each target parameter to save for dynamic clipping
    bounds = {}
    for col in target_cols:
        bounds[col] = {
            'min': float(y[col].min()),
            'max': float(y[col].max())
        }
    bounds['dw_mm'] = {
        'min': float(df['dw_mm'].min()),
        'max': float(df['dw_mm'].max())
    }
    
    print("\nCalculated Target Parameter Bounds:")
    for col, b in bounds.items():
        print(f"  {col:<12} -> Min: {b['min']:6.1f}, Max: {b['max']:6.1f}")
        
    # 2. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # 3. Define pipelines with StandardScaler
    models = {
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(Ridge(alpha=1.0)))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'Extra Trees': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)))
        ])
    }
    
    # 4. Evaluate Models
    best_model_name = None
    best_avg_r2 = -float('inf')
    model_eval_results = {}
    
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        param_metrics = {}
        total_r2 = 0
        for idx, col in enumerate(target_cols):
            mae = mean_absolute_error(y_test.iloc[:, idx], preds[:, idx])
            r2 = r2_score(y_test.iloc[:, idx], preds[:, idx])
            param_metrics[col] = {'MAE': mae, 'R2': r2}
            total_r2 += r2
            
        avg_r2 = total_r2 / len(target_cols)
        model_eval_results[name] = (pipeline, param_metrics, avg_r2)
        print(f"\nModel: {name} | Average R2 score: {avg_r2:.4f}")
        for col, m in param_metrics.items():
            print(f"  {col:<12} -> MAE: {m['MAE']:8.4f}, R2: {m['R2']:8.4f}")
            
        if avg_r2 > best_avg_r2:
            best_avg_r2 = avg_r2
            best_model_name = name

    print(f"\n--> Selected Best Model: {best_model_name}")
    
    # 5. Train the best model on full data
    final_pipeline = models[best_model_name]
    final_pipeline.fit(X, y)
    
    # 6. Save model and metadata
    model_data = {
        'model': final_pipeline,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'bounds': bounds,
        'mappings': {
            'footpath': {'None': 0, 'Single Side': 1, 'Both Sides': 2},
            'steel_grade': {'E 350A': 350, 'E 410A': 410, 'E 450A': 450},
            'concrete_grade': {'M40': 40, 'M50': 50, 'M60': 60}
        },
        'model_name': best_model_name
    }
    
    os.makedirs('src', exist_ok=True)
    model_pkl_path = r"d:/Projects/bridge-optimization/src/model.pkl"
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel successfully saved to {model_pkl_path}")
    
    # 7. Generate diagnostic plots (True vs Predicted scatter plots)
    # Get predictions on the test set from the final pipeline
    test_preds = final_pipeline.predict(X_test)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    
    for idx, col in enumerate(target_cols):
        true_vals = y_test.iloc[:, idx]
        pred_vals = test_preds[:, idx]
        
        # Plot scatter
        axes[idx].scatter(true_vals, pred_vals, alpha=0.5, color='blue', edgecolor='k')
        
        # Plot y=x diagonal line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], '--', color='red', linewidth=2)
        
        # Titles and labels
        metrics = model_eval_results[best_model_name][1][col]
        axes[idx].set_title(f"{col}\nR2: {metrics['R2']:.4f} | MAE: {metrics['MAE']:.3f}")
        axes[idx].set_xlabel("True Values")
        axes[idx].set_ylabel("Predicted Values")
        axes[idx].grid(True)
        
    # Hide empty subplots
    for i in range(len(target_cols), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plot_path = r"D:\Projects\bridge-optimization\diagnostic_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Diagnostic True vs Predicted plots saved to {plot_path}")
    plt.close()

if __name__ == '__main__':
    main()