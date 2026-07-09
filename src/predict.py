import os
import pickle
import argparse
import numpy as np
import pandas as pd

def compute_weights(span, n, carriageway_width, footpath_encoded, t_slab, bf, tf, tw, dw,
                    include_median, median_width, footpath_width, railing_width, crash_barrier_width):
    """
    Computes exact structural weights (kN) and overall widths using physical formulas.
    """
    no_of_carriageways = 2 if include_median else 1
    w_median = median_width if include_median else 0.0
    
    # Overall Bridge Width (Nominal & Effective based on your updated formula)
    overall_width = (no_of_carriageways * carriageway_width) + w_median + footpath_encoded * (footpath_width + railing_width) + 2.0 * crash_barrier_width
    width_eff = overall_width
    
    # Concrete deck weight: volume * concrete density (25.0 kN/m^3)
    deck_weight = span * (t_slab / 1000.0) * width_eff * 25.0
    
    # Steel girders weight: volume * steel density (78.5 kN/m^3)
    area_steel = (2.0 * bf * tf + dw * tw) / 1000000.0
    steel_weight = n * span * area_steel * 78.5
    
    total_weight = steel_weight + deck_weight
    return steel_weight, deck_weight, total_weight, overall_width

def main():
    parser = argparse.ArgumentParser(description="Predict optimal bridge parameters given span length and custom inputs.")
    parser.add_argument("--span", type=float, required=True, help="Span length of the bridge in meters")
    parser.add_argument("--width", type=float, required=True, help="Carriageway width in meters")
    parser.add_argument("--footpath", type=str, default="None", choices=["None", "Single Side", "Both Sides"], help="Footpath configuration (default: None)")
    parser.add_argument("--skew", type=float, default=0.0, help="Skew angle in degrees (default: 0.0)")
    parser.add_argument("--steel", type=str, default="E 350A", choices=["E 350A", "E 410A", "E 450A"], help="Steel grade (default: E 350A)")
    parser.add_argument("--concrete", type=str, default="M40", choices=["M40", "M50", "M60"], help="Concrete grade (default: M40)")
    
    # Options for widths to fit the user's overall width formula
    parser.add_argument("--median_width", type=float, default=1.2, help="Width of the median in meters if present (default: 1.2)")
    parser.add_argument("--footpath_width", type=float, default=1.5, help="Width of the footpath in meters (default: 1.5)")
    parser.add_argument("--railing_width", type=float, default=0.375, help="Width of the railing in meters (default: 0.375)")
    parser.add_argument("--crash_barrier_width", type=float, default=0.45, help="Width of the crash barrier in meters (default: 0.45)")
    
    args = parser.parse_args()
    
    model_pkl_path = r"d:/Projects/bridge-optimization/src/model.pkl"
    if not os.path.exists(model_pkl_path):
        print(f"Error: Model file {model_pkl_path} not found. Please run src/train_model.py first.")
        return
        
    with open(model_pkl_path, 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    target_cols = model_data['target_cols']
    bounds = model_data['bounds']
    mappings = model_data['mappings']
    
    # Check if median is present based on dataset mapping
    include_median_bool = False
    median_str = "No"
    if args.median_width > 0 and args.width > 8.0: 
        pass
        
    print(f"\n--- Bridge Parameter Predictor (Model: {model_data['model_name']}) ---")
    print("Inputs:")
    print(f"  Span Length:         {args.span:.2f} m")
    print(f"  Carriageway Width:   {args.width:.2f} m")
    print(f"  Include Footpath:    {args.footpath}")
    print(f"  Skew Angle:          {args.skew:.2f} deg")
    print(f"  Steel Grade:         {args.steel}")
    print(f"  Concrete Grade:      {args.concrete}")
    print(f"  Median Width:        {args.median_width:.2f} m")
    print(f"  Footpath Width:      {args.footpath_width:.2f} m")
    print(f"  Railing Width:       {args.railing_width:.2f} m")
    print(f"  Crash Barrier Width: {args.crash_barrier_width:.2f} m")
    
    # 1. Preprocess input
    footpath_val = mappings['footpath'].get(args.footpath, 0)
    steel_val = mappings['steel_grade'].get(args.steel, 350)
    concrete_val = mappings['concrete_grade'].get(args.concrete, 40)
    
    # Note: include_median in the dataset was always 'No' (mapped to 0)
    input_data = pd.DataFrame([{
        'span_m': args.span,
        'carriageway_width_m': args.width,
        'footpath_encoded': footpath_val,
        'skew_angle_deg': args.skew,
        'steel_grade_encoded': steel_val,
        'concrete_grade_encoded': concrete_val
    }])
    
    # Ensure columns order matches the model training features order
    input_data = input_data[feature_cols]
    
    # 2. Predict raw values
    raw_preds = model.predict(input_data)[0]
    preds_dict = dict(zip(target_cols, raw_preds))
    
    # 3. Post-process to ensure physical validity and dynamic dataset bounds
    n_opt = int(np.clip(round(preds_dict['n']), bounds['n']['min'], bounds['n']['max']))
    spacing_opt = float(np.clip(preds_dict['spacing_m'], bounds['spacing_m']['min'], bounds['spacing_m']['max']))
    t_slab_opt = float(np.clip(round(preds_dict['t_slab_mm']), bounds['t_slab_mm']['min'], bounds['t_slab_mm']['max']))
    D_opt = float(np.clip(round(preds_dict['D_mm']), bounds['D_mm']['min'], bounds['D_mm']['max']))
    bf_opt = float(np.clip(round(preds_dict['bf_mm']), bounds['bf_mm']['min'], bounds['bf_mm']['max']))
    tf_opt = float(np.clip(round(preds_dict['tf_mm']), bounds['tf_mm']['min'], bounds['tf_mm']['max']))
    tw_opt = float(np.clip(round(preds_dict['tw_mm']), bounds['tw_mm']['min'], bounds['tw_mm']['max']))
    
    # Compute web depth dw geometrically: dw = D - 2*tf
    dw_opt = D_opt - 2.0 * tf_opt
    
    # Clip dw to bounds and adjust depth if needed
    dw_min = bounds['dw_mm']['min']
    dw_max = bounds['dw_mm']['max']
    if dw_opt < dw_min:
        dw_opt = dw_min
        D_opt = dw_opt + 2.0 * tf_opt
    elif dw_opt > dw_max:
        dw_opt = dw_max
        D_opt = dw_opt + 2.0 * tf_opt

    # 4. Compute physical weights and overall widths
    steel_w, deck_w, total_w, overall_width = compute_weights(
        args.span, n_opt, args.width, footpath_val, t_slab_opt, bf_opt, tf_opt, tw_opt, dw_opt,
        include_median_bool, args.median_width, args.footpath_width, args.railing_width, args.crash_barrier_width
    )
    
    # 5. Output results
    print("\nPredicted Optimal Bridge Parameters:")
    print(f"  Number of Girders (n):         {n_opt}")
    print(f"  Girder Spacing (spacing_m):    {spacing_opt:.3f} m")
    print(f"  Deck Slab Thickness (t_slab):  {t_slab_opt:.1f} mm")
    print(f"  Overall Girder Depth (D):      {D_opt:.1f} mm")
    print(f"  Flange Width (bf):             {bf_opt:.1f} mm")
    print(f"  Flange Thickness (tf):         {tf_opt:.1f} mm")
    print(f"  Web Thickness (tw):            {tw_opt:.1f} mm")
    print(f"  Web Depth (dw):                {dw_opt:.1f} mm  (computed: D - 2*tf)")
    
    print("\nEstimated Bridge Geometry & Weights:")
    print(f"  Overall Bridge Width:          {overall_width:.2f} m  (based on your formula)")
    print(f"  Concrete Deck Weight:          {deck_w:.2f} kN")
    print(f"  Structural Steel Weight:       {steel_w:.2f} kN")
    print(f"  Total Structural Weight:       {total_w:.2f} kN")
    print("--------------------------------------------------")

if __name__ == '__main__':
    main()
