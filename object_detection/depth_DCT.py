# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

import argparse
import os
from collections import Counter

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from depth_parent import eral
from matplotlib import cm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from scipy.interpolate import UnivariateSpline
from scipy.stats import pearsonr
from skopt import gp_minimize

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['font.size'] = 10
ADD_PATH = "/path/to/data/"


def run_function(dataset="kitti", yolo=False, val_activate=False):
    """Run the evaluation and analysis for depth detection relationships.
    Args:
        dataset (str): Dataset to use, e.g., "kitti", "nuscenes".
        val_activate (bool): Whether to use validation set or not.
        yolo (bool): Whether to use YOLO for object detection or not.
    """

    
    print(f"Dataset: {dataset}")
    print(f"YOLO: {yolo}")
    print(f"Validation activated: {val_activate}")
    model_name = f"EXP_{dataset.upper()}_depth_detection_inverse_minmax_expo_teacher10_V0"
    if val_activate:
        plot_addition_val = "val"
        depth_maps_folder = ADD_PATH+f"/DepthPrior/datasets/{dataset}/depth_validation/"  # update this to your actual folder path
    else:
        plot_addition_val = "infer"
        depth_maps_folder = ADD_PATH+f"/DepthPrior/datasets/{dataset}/depth_anything_predictions/"  # update this to your actual folder path

    if yolo:
        if val_activate: infer_val = "val"
        else: infer_val = "infer"
        if dataset == "KITTI":
            model_name_yolo = "kitti"
        elif dataset == "visdrone":
            model_name_yolo = "visdrone"
        elif dataset == "MSCOCO":
            model_name_yolo = "coco"
        elif dataset == "SUNRGBD":
            model_name_yolo = "sunrgbd2d"

        yolo_predictions_path = ADD_PATH+f"/DepthPrior/object_detection/yolov11/runs/{model_name_yolo}-subset/detect/train/predictions_{infer_val}.jsonl"
        plot_addition=f"yolov11/{dataset}/{plot_addition_val}"
    else:
        yolo_predictions_path = None
        plot_addition=f"efficientdet/{dataset}/{plot_addition_val}"

    os.makedirs(ADD_PATH+f"/DepthPrior/object_detection/plots_DCT/{plot_addition}", exist_ok=True)
    print(f"Saving plots to {ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}'}")
    eval_relax = eral(dataset=dataset,
                        model_name=model_name, # only relevant for kitti
                        valdidation_activate=val_activate,                     
                        depth_maps_folder=depth_maps_folder,
                        yolo_path=yolo_predictions_path)

    eval_relax.eval_threshold_relaxation(score_thr=0.1, lift_function = None)
    thr_vals = np.round(np.linspace(0.1, 0.9, 9),2)
    depth_knots = np.round(np.linspace(0, 0.9, 10),2)
    if val_activate:
        def optimize_spline_pareto_dominance(eval_relax, tau, depth_knots, all_static_baselines):
            """
            Simple, aggressive optimization with Pareto dominance constraint.
            No pre-calculations, full exploration range.
            """
            tp_static, fp_static = all_static_baselines[tau]            
            # Define the allowed FP range - stay in same precision regime
            fp_tolerance = 0.1  # 10% tolerance
            fp_max = fp_static + max(1, int(fp_static * fp_tolerance))
            print("Max FP Tolerance", fp_max-fp_static)
            
            def objective(spline_values):
                try:
                    # Create the spline
                    threshold_spline = UnivariateSpline(depth_knots, spline_values, k=3, s=0.1)

                    # Define the lift function
                    def lift_function(depth):
                        return tau - np.clip(threshold_spline(depth), 0.0, 1.0)

                    # Evaluate the thresholds
                    tp, fp = eval_relax.eval_threshold_relaxation(score_thr=tau, lift_function=lift_function)

                    # Handle invalid values for tp or fp
                    if tp is None or fp is None or np.isnan(tp) or np.isnan(fp) or np.isinf(tp) or np.isinf(fp):
                        print(f"Invalid TP/FP values: TP={tp}, FP={fp}")
                        return 1e6  # Large penalty for invalid values

                    # Debug: Print every evaluation
                    print(f"Eval: TP={tp:.0f}, FP={fp:.0f}, fp_max={fp_max:.1f}")

                    # Calculate improvements and penalties
                    tp_improvement = tp - tp_static
                    fp_worsening = fp - fp_static

                    # Apply constraints
                    if tp_improvement < 0:
                        print(f"REJECTED: TP improvement {tp_improvement} < 0")
                        return 1e3 * (-tp_improvement)  # Penalize negative TP improvement
                    elif fp > fp_max:
                        print(f"REJECTED: FP {fp} > {fp_max}")
                        return 1e3 * (fp - fp_max)  # Penalize exceeding FP tolerance
                    elif tp_improvement <= fp_worsening:
                        print(f"REJECTED: No sufficient TP improvement")
                        return 1e3 * (fp_worsening - tp_improvement)  # Penalize insufficient improvement
                    else:
                        print(f"ACCEPTED: TP improvement = {tp_improvement}")
                        # FIX: Avoid division by zero and ensure finite values
                        if fp > 0:
                            ratio_term = tp / fp #- tp_static / fp_static
                        else:
                            ratio_term = 0  # or use a large value like 1e6
                        
                        objective_value = -tp_improvement + fp_worsening - ratio_term
                        
                        # Safety check: ensure the return value is finite
                        if not np.isfinite(objective_value):
                            print(f"WARNING: Non-finite objective value, returning penalty")
                            return 1e6
                            
                        return objective_value

                except Exception as e:
                    print(f"ERROR in objective function: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1e6  # Large penalty for exceptions
            
            bounds = [(0.1, 1.0) for _ in depth_knots]
            initial_spline_values = [tau] * len(depth_knots)
            
            # Try Bayesian optimization with additional safety
            result = gp_minimize(objective, bounds, n_calls=150, random_state=42, x0=initial_spline_values)            
            return result.x
        # Step 1: Collect all static baselines once
        all_static_baselines = {}
        for tau in thr_vals:
            tp, fp = eval_relax.eval_threshold_relaxation(score_thr=tau, lift_function=None)
            all_static_baselines[tau] = (tp, fp)

        # Extract (fp, tp) pairs and sort by FP
        fp_tp_pairs = [(fp, tp) for tp, fp in all_static_baselines.values()]
        fp_tp_pairs.sort(key=lambda x: x[0])  # Sort by FP

        # Step 2: Optimize each threshold independently
        results = {}
        for tau in thr_vals[::-1]:
            print(f"Optimizing alternative to static threshold {tau}...")
            best_spline = optimize_spline_pareto_dominance(eval_relax, tau, depth_knots, all_static_baselines)
            # Test the result
            spline = UnivariateSpline(depth_knots, best_spline, k=3, s=0.1)
            def lift_fn(depth): return tau - np.clip(spline(depth), 0.0, 1.0)
            tp_opt, fp_opt = eval_relax.eval_threshold_relaxation(score_thr=tau, lift_function=lift_fn)
            if (tp_opt < all_static_baselines[tau][0] and fp_opt > all_static_baselines[tau][1]) or (tp_opt - all_static_baselines[tau][0] < fp_opt - all_static_baselines[tau][1]):
                print(f"Optimization failed for tau={tau}: TP {tp_opt} vs {all_static_baselines[tau][0]}, FP {fp_opt} vs {all_static_baselines[tau][1]}")
                tp_opt = all_static_baselines[tau][0]
                fp_opt = all_static_baselines[tau][1]
                best_spline = [tau] * len(depth_knots)  # Fallback to static threshold
            results[tau] = {
                'spline_params': best_spline,
                'improvement': f"TP {all_static_baselines[tau][0]}→{tp_opt}, FP {all_static_baselines[tau][1]}→{fp_opt}"
            }
        # Step 3: Save lookup table
        def save_spline_lookup_table(results, depth_knots, all_static_baselines, model_name):
            """Save spline lookup table with static baseline values included."""

            lookup_file = ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/threshold_lookup.txt'

            with open(lookup_file, 'w') as f:
                f.write("# Spline-based depth-adaptive thresholds lookup table\n")
                f.write("# Format: static_threshold, static_tp, static_fp, optimized_tp, optimized_fp, strategy, reduction_range, depth_knots, spline_params\n")
                f.write(f"# Depth knots: {depth_knots.tolist()}\n")
                f.write("# Usage: Create spline from params, apply as lift function\n\n")
                
                for tau, data in results.items():
                    # Get static baseline values
                    static_tp, static_fp = all_static_baselines[tau]
                    
                    # Extract improved TP and FP from improvement string
                    improvement = data['improvement']
                    # Parse "TP 2169→3836, FP 7→29" format
                    tp_part = improvement.split('→')[1].split(',')[0]
                    fp_part = improvement.split('→')[2]
                    
                    improved_tp = int(tp_part)
                    improved_fp = int(fp_part)
                    
                    # Write: threshold, static_tp, static_fp, improved_tp, improved_fp, knots, params
                    f.write(f"{tau:.2f}, {static_tp}, {static_fp}, {improved_tp}, {improved_fp}, ")
                    f.write(f"{depth_knots.tolist()}, {data['spline_params']}\n")
            
            print(f"Spline lookup table saved to: {lookup_file}")

        # Save your results
        save_spline_lookup_table(results, depth_knots, all_static_baselines, f"{plot_addition}")

        # Example usage
        lookup_path = ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/threshold_lookup.txt'
        output_dir = ADD_PATH + f"/DepthPrior/object_detection/plots_DCT/{plot_addition}/"
        # Read the lookup table
        results = []
        with open(lookup_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Parse: static_threshold, static_tp, static_fp, optimized_tp, optimized_fp, depth_knots, spline_params
                parts = line.split(", ", 8)
                static_thr = float(parts[0])
                static_tp = int(parts[1])
                static_fp = int(parts[2])
                optimized_tp = int(parts[3])
                optimized_fp = int(parts[4])
                depth_knots_and_spline = str(parts[5:]).strip("[]")  # Remove the outer brackets
                depth_knots, spline_params = depth_knots_and_spline.split("], [")  # Split into two parts
                # Convert the strings back to lists
                depth_knots = eval(str(depth_knots[2:].replace("'", "")))
                spline_params = eval(spline_params[:-2])
                results.append({
                    "tau_0": static_thr,
                    "static TD": static_tp,
                    "static ED": static_fp,
                    "optimized TD": optimized_tp,
                    "optimized ED": optimized_fp,
                    "depth_knots": depth_knots,
                    "spline_params": spline_params
                })

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        latex_content = []
        latex_content.append(r"\begin{tabular}{l|" + "c" * len(depth_knots) + "}")
        latex_content.append(r"\hline")
        latex_content.append(r"& \multicolumn{" + str(len(depth_knots)) + r"}{c}{Depth Knots} \\")
        latex_content.append(r"$\tau_0$ & " + " & ".join([f"{knot:.2f}" for knot in depth_knots]) + r" \\")
        latex_content.append(r"\cline{2-" + str(len(depth_knots) + 1) + r"}")
        latex_content.append(r"& \multicolumn{" + str(len(depth_knots)) + r"}{c}{Spline Parameters} \\")
        latex_content.append(r"\hline")

        # Add rows for each tau_0 and corresponding spline parameters
        for _, row in df.iterrows():
            tau_0 = f"{row['tau_0']:.2f}"
            spline_params_str = " & ".join([f"{param:.2f}" for param in row["spline_params"]])
            latex_content.append(f"{tau_0} & {spline_params_str} \\\\")

        latex_content.append(r"\hline")
        latex_content.append(r"\end{tabular}")

        # Save LaTeX table to file
        output_dir = ADD_PATH + f'/DepthPrior/object_detection/plots_DCT/{plot_addition}'
        depth_knots_table_path = f"{output_dir}/depth_knots_table.tex"
        with open(depth_knots_table_path, "w") as f:
            f.write("\n".join(latex_content))

        print(f"Depth knots table saved to: {depth_knots_table_path}")

        # Save LaTeX table for performance comparison
        performance_table = df[["tau_0", "static TD", "static ED", "optimized TD", "optimized ED"]].copy()
        performance_table[r"$\Delta_{TD}$"] = performance_table["optimized TD"] - performance_table["static TD"]
        performance_table[r"$\Delta_{ED}$"] = performance_table["optimized ED"] - performance_table["static ED"]
        # Rename the tau_0 column to have LaTeX formatting
        performance_table = performance_table.rename(columns={"tau_0": r"$\tau_0$"})
        # Reorder columns to have tau_0 first
        column_order = [r"$\tau_0$", "static TD", "static ED", "optimized TD", "optimized ED", r"$\Delta_{TD}$", r"$\Delta_{ED}$"]
        performance_table = performance_table[column_order]
        # Save to LaTeX
        performance_table_path = f"{output_dir}/performance_table.tex"
        performance_table.to_latex(performance_table_path, index=False, float_format="%.2f", escape=False)
        print(f"Performance table saved to: {performance_table_path}")


        # Plot Pareto front
        plt.figure(figsize=(4, 2.5), dpi=100)
        tab10_colors = plt.cm.tab10.colors

        # Plot static and optimized Pareto fronts
        plt.plot(df["static ED"], df["static TD"], 'o--', label="Static", color=tab10_colors[0], linewidth=1, markersize=2)
        plt.plot(df["optimized ED"], df["optimized TD"], 's-', label="DCT", color=tab10_colors[1], linewidth=1, markersize=2)

        # Add arrows showing improvements
        for i in range(len(df)):
            plt.annotate('', xy=(df["optimized ED"][i], df["optimized TD"][i]), 
                        xytext=(df["static ED"][i], df["static TD"][i]),
                        arrowprops=dict(arrowstyle='->', color=tab10_colors[2], alpha=0.7, lw=1))

        # Add labels for thresholds
        for i, tau in enumerate(df["tau_0"]):
            plt.text(df["static ED"][i], df["static TD"][i], f"{tau:.1f}", fontsize=8, color=tab10_colors[0], ha="right")
            plt.text(df["optimized ED"][i], df["optimized TD"][i], f"{tau:.1f}", fontsize=8, color=tab10_colors[1], ha="left")

        plt.xlabel("EDs")
        plt.ylabel("TDs")
        plt.legend(loc='best', ncol=2, labelspacing=0.0,
                columnspacing=0.3, markerscale=1.5, handlelength=1.3, handletextpad=0.2)
        plt.grid(True, alpha=0.3)

        # Set log scales
        plt.xscale("log")
        plt.yscale("log")

        # Set reasonable axis limits
        x_min = min(df["static ED"].min(), df["optimized ED"].min())
        x_max = max(df["static ED"].max(), df["optimized ED"].max())
        y_min = min(df["static TD"].min(), df["optimized TD"].min())
        y_max = max(df["static TD"].max(), df["optimized TD"].max())

        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1

        plt.xlim(max(1, x_min - x_padding), x_max + x_padding)
        plt.ylim(max(1, y_min - y_padding), y_max + y_padding)

        # Add improvement statistics
        total_tp_gain = sum(df["optimized TD"]) - sum(df["static TD"])
        total_fp_change = sum(df["optimized ED"]) - sum(df["static ED"])

        stats_text = f'+{total_tp_gain:.0f} TDs\n'
        stats_text += f'{total_fp_change:+.0f} EDs'

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Use bbox_inches=None instead of "tight" to avoid recalculation
        plt.savefig(f"{output_dir}/pareto_front_read.png", dpi=150)  # Reduced DPI
        plt.savefig(f"{output_dir}/pareto_front_read.svg")  # SVG doesn't have DPI issues
        plt.close()

    else:
        lookup_path = ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/threshold_lookup.txt'
        lookup_path = lookup_path.replace("infer", "val")  # Use validation lookup table
        splines_data = {}

        with open(lookup_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Parse line
                parts = line.split(", ", 8)
                static_thr = float(parts[0])
                spline_params = eval(parts[8].split("[")[-1][:-1])  # Convert string back to list 
                
                splines_data[static_thr] = spline_params

        static_threshold=0.7
        # Load spline for adaptive threshold
        spline = UnivariateSpline(depth_knots, splines_data[static_threshold], k=3, s=0.1)
        
        # Find the image with the most new detections
        max_new_detections = 0
        image_idx = 0  # Default to the first image if no new detections are found

        for idx in range(len(eval_relax.collect_pseudo_boxes)):
            pred_boxes = eval_relax.collect_pseudo_boxes[idx]
            pred_scores = eval_relax.collect_pseudo_scores[idx]
            pred_depths = eval_relax.collect_depths[idx]
            
            new_detections_count = 0
            for box, score, depth in zip(pred_boxes, pred_scores, pred_depths):
                adaptive_threshold = np.clip(spline(depth), 0.0, 1.0)
                adjusted_score = score + (static_threshold - adaptive_threshold)
                if adjusted_score >= static_threshold and score < static_threshold:
                    new_detections_count += 1
            
            if new_detections_count > max_new_detections:
                max_new_detections = new_detections_count
                image_idx = idx

        print(f"Image with the most new detections: {image_idx} (New Detections: {max_new_detections})")


        # Get predictions for this image
        pred_boxes = eval_relax.collect_pseudo_boxes[image_idx]
        pred_scores = eval_relax.collect_pseudo_scores[image_idx]
        pred_depths = eval_relax.collect_depths[image_idx]  # Fixed: using collect_depths
        pred_classes = eval_relax.collect_pseudo_classes[image_idx]

        # Try to load actual image if available
        image_name = eval_relax.images_data[image_idx].replace('.txt', f'.{eval_relax.im_format}')
        image_path = os.path.join(eval_relax.gt_images_folder, image_name)

        # Use tab10 colormap
        tab10_colors = cm.tab10.colors
        color_existing = tab10_colors[2]  # Green for existing detections
        color_new = tab10_colors[1]       # Orange for new detections


        # Track detections and label positions
        static_detections = 0
        adaptive_detections = 0
        new_detections = []
        label_positions = []  # Track positions to avoid overlaps

        # First pass: collect all detection info
        detection_info = []
        for i, (box, score, depth) in enumerate(zip(pred_boxes, pred_scores, pred_depths)):
            y1, x1, y2, x2 = box
            adaptive_threshold = np.clip(spline(depth), 0.0, 1.0)
            adjusted_score = score + (static_threshold - adaptive_threshold)    
            if adjusted_score >= static_threshold:
                is_new = score < static_threshold
                detection_info.append({
                    'box': box,
                    'score': score,
                    'threshold': adaptive_threshold,
                    'is_new': is_new,
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'area': (x2 - x1) * (y2 - y1)
                })        
                adaptive_detections += 1
                if is_new:
                    new_detections.append(i)
                else:
                    static_detections += 1

        # Sort by area (larger boxes first) to draw them in proper order
        detection_info.sort(key=lambda x: x['area'], reverse=True)

        # Set up figure with better proportions for single image
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        # Load and display image
        if os.path.exists(image_path):
            img = plt.imread(image_path)
            ax.imshow(img)
        else:
            ax.set_xlim(0, 1200)
            ax.set_ylim(400, 0)

        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Draw detections
        for det in detection_info:
            y1, x1, y2, x2 = det['box']
            width = x2 - x1
            height = y2 - y1    
            # Choose color based on detection type
            box_linewidth = 2
            box_alpha = 1.0
            if det['is_new']:
                edge_color = color_new
            else:
                edge_color = color_existing
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), width, height, 
                                    linewidth=box_linewidth, 
                                    edgecolor=edge_color, 
                                    facecolor='none', 
                                    alpha=box_alpha)
            ax.add_patch(rect)    
            # Prepare label text
            label = f"{det['score']:.2f}\n(τ={det['threshold']:.2f})"  # Include τ for both types of detections
            
            # Find good label position (avoid overlaps)
            label_x = x1 + width / 2
            label_y = y1 - 10  # Default: above box    
            # Check for overlaps and adjust position
            for prev_pos in label_positions:
                if abs(label_x - prev_pos[0]) < 50 and abs(label_y - prev_pos[1]) < 20:
                    label_y = y2 + 20  # Move to below box if overlap
                    break    
            label_positions.append((label_x, label_y))    
            # Draw label with better visibility
            bbox_props = dict(
                facecolor=edge_color,
                linewidth=1,
                alpha=0.5
            )    
            ax.text(label_x, label_y, label, 
                    fontsize=10,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va='center' if label_y > y2 else 'bottom',
                    bbox=bbox_props,)  
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='none', edgecolor=color_existing, linewidth=2, 
                            label=r'Static $\tau_0=$' + f'{static_threshold:.2f} ({static_detections})'),
            patches.Patch(facecolor='none', edgecolor=color_new, linewidth=2,
                            label=f'DCT (+{len(new_detections)})')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        # Save with high quality for publication
        plt.savefig(ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/academic_example_{image_idx}.png',
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/academic_example_{image_idx}.svg',
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"Example image {image_idx}: Base={static_detections}, New=+{len(new_detections)}, Total={adaptive_detections}")


        # Create depth range for plotting
        depth_range = np.linspace(0, 1, 200)

        # 1) Plot all spline functions together
        plt.figure(figsize=(2.5, 1.5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(splines_data)))
        linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', '<', '>']
        plt.plot([], [], color='gray', linestyle='--', label='Static')

        for i, (tau, params) in enumerate(sorted(splines_data.items())):
            color = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)]
            mk = markers[i % len(markers)]
            # Create spline
            spline = UnivariateSpline(depth_knots, params, k=3, s=0.1)
            threshold_curve = np.clip(spline(depth_range), 0.0, 1.0)
            line_artist, = plt.plot(depth_range, threshold_curve, marker=mk, color=color, markersize=2, markevery=20, linestyle=ls,
                                    linewidth=1, alpha=0.9, label=r'$\tau_0=$' + f'{tau:.1f}')
            # Plot horizontal static threshold (no label; will be included in combined legend entry)
            hline_artist = plt.axhline(y=tau, color=color, linestyle='--', linewidth=1, alpha=0.8)

        plt.xlabel(r'Depth $d$ (0=close, 1=far)')
        plt.ylabel(r'$\tau$')

        # Use HandlerTuple so each label shows both the spline style and the static horizontal style
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, labelspacing=0.0,
                columnspacing=0.3, markerscale=1.5, handlelength=1.3, handletextpad=0.2,)

        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/all_spline_functions.png', bbox_inches='tight')
        plt.savefig(ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/all_spline_functions.svg', bbox_inches='tight')
        plt.savefig(ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/all_spline_functions.pdf', bbox_inches='tight')
        plt.close()

        # Initialize arrays for plotting
        static_original_pts = []
        depth_aware_pts = []
        spline_results = []

        print("Loading and evaluating spline lookup table...")
        print("="*60)

        with open(lookup_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue       
                # Parse: static_threshold, static_tp, static_fp, improved_tp, improved_fp, strategy, reduction_range, depth_knots, spline_params
                parts = line.split(", ", 8)
                static_thr = float(parts[0])
                # Extract spline params from the end of the line
                spline_params = eval(parts[8].split("[")[-1][:-1])  # Convert string back to list       
                # Add static point
                static_tp, static_fp = eval_relax.eval_threshold_relaxation(
                    score_thr=static_thr, lift_function=None)     
                static_original_pts.append((static_fp, static_tp, static_thr))
                # if static_thr == area_comparison_tau: static_data = collect_areas(eval_relax)
                
                # Create spline and test
                spline = UnivariateSpline(depth_knots, spline_params, k=3, s=0.1)
                def lift_fn(depth):
                    dynamic_thresh = np.clip(spline(depth), 0.0, 1.0)
                    return static_thr - dynamic_thresh       
                # Evaluate spline result
                tp_test, fp_test = eval_relax.eval_threshold_relaxation(
                    score_thr=static_thr, lift_function=lift_fn)    
                # if static_thr == area_comparison_tau: depth_data = collect_areas(eval_relax)   
                # Add depth-aware point
                depth_aware_pts.append((fp_test, tp_test, static_thr))       
                spline_results.append({
                    '$\\tau_0$': static_thr,
                    'static TD': static_tp,
                    'static ED': static_fp,
                    'optimized TD': tp_test,
                    'optimized ED': fp_test,
                    r"$\Delta_{TD}$": tp_test - static_tp,
                    r"$\Delta_{ED}$": fp_test - static_fp
                })       
                print(f"Threshold {static_thr:.2f}: Static TP={static_tp:4d}, FP={static_fp:4d} | "
                        f"Optimized TP={tp_test:4d} ({tp_test-static_tp:+3d}), FP={fp_test:4d} ({fp_test-static_fp:+3d})")

        # plot_size_comparison(static_data, depth_data, area_comparison_tau)
        print("="*60)
        print("Saving spline results to file...")

        # Define the output path for the results file
        results_path = ADD_PATH+f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/spline_results.json'
        print(f"Results saved to {results_path}")
        df = pd.DataFrame(spline_results)
        df.to_csv(results_path, index=False)

        # Convert to LaTeX table
        latex_table = df.to_latex(index=False, float_format="%.2f")

        # Save to TXT file
        latex_table_path = ADD_PATH + f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/spline_results_table.tex'
        with open(latex_table_path, "w") as f:
            f.write(latex_table)

        print(f"LaTeX table saved to: {latex_table_path}")


        # !import code; code.interact(local=vars())
        # static_threshold=0.4
        # Load spline for adaptive threshold
        spline = UnivariateSpline(depth_knots, splines_data[static_threshold], k=3, s=0.1)

        # Collect statistics
        static_stats = {'sizes': [], 'depths': [], 'classes': [], 'boxes': []}
        adaptive_stats = {'sizes': [], 'depths': [], 'classes': [], 'boxes': []}
        new_detection_stats = {'sizes': [], 'depths': [], 'classes': []}

        for img_idx in range(min(len(eval_relax.collect_pseudo_boxes), 1000)):  # Limit to 1000 images for performance
            pred_boxes = eval_relax.collect_pseudo_boxes[img_idx]
            pred_scores = eval_relax.collect_pseudo_scores[img_idx]
            pred_depths = eval_relax.collect_depths[img_idx]  # Fixed: using collect_depths
            pred_classes = eval_relax.collect_pseudo_classes[img_idx]    
            for box, score, depth, cls in zip(pred_boxes, pred_scores, pred_depths, pred_classes):
                # box format is [y1, x1, y2, x2]
                area = (box[2] - box[0]) * (box[3] - box[1])        
                # Static threshold
                passes_static = score >= static_threshold        
                # Adaptive threshold
                adaptive_threshold = np.clip(spline(depth), 0.0, 1.0)
                passes_adaptive = score >= adaptive_threshold        
                if passes_static:
                    static_stats['sizes'].append(area)
                    static_stats['depths'].append(depth)
                    static_stats['classes'].append(cls)
                    static_stats['boxes'].append(box)        
                if passes_adaptive:
                    adaptive_stats['sizes'].append(area)
                    adaptive_stats['depths'].append(depth)
                    adaptive_stats['classes'].append(cls)
                    adaptive_stats['boxes'].append(box)            
                    # Track new detections
                    if not passes_static:
                        new_detection_stats['sizes'].append(area)
                        new_detection_stats['depths'].append(depth)
                        new_detection_stats['classes'].append(cls)


        fig = plt.figure(figsize=(4.5,3))
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.4)
        tab10_colors = plt.cm.tab10.colors

        # 1. Size distribution comparison
        ax1 = fig.add_subplot(gs[0, 0])
        if len(static_stats['sizes']) > 0 and len(adaptive_stats['sizes']) > 0:
            sizes_static = np.log10(np.array(static_stats['sizes']) + 1)
            sizes_adaptive = np.log10(np.array(adaptive_stats['sizes']) + 1)
            bins = np.linspace(min(sizes_static.min(), sizes_adaptive.min()),
                            max(sizes_static.max(), sizes_adaptive.max()), 10)
            ax1.hist(sizes_adaptive, bins=bins, alpha=0.7, label=f'DCT',
                    color=tab10_colors[1], density=True)
            ax1.hist(sizes_static, bins=bins, alpha=0.7, label=f'Static',
                    color=tab10_colors[0], density=True)
            if len(new_detection_stats['sizes']) > 0:
                sizes_new = np.log10(np.array(new_detection_stats['sizes']) + 1)
                ax1.hist(sizes_new, bins=bins, alpha=0.7, 
                        color=tab10_colors[1], density=True, histtype='step', linewidth=1.5)

        ax1.set_xlabel(r'$\log_{10}$(Area + 1)')
        ax1.set_ylabel('Density')
        ax1.legend(labelspacing=0.0,columnspacing=0.3, markerscale=1.5, handlelength=1.3, handletextpad=0.2,)
        ax1.grid(True, alpha=0.3)

        # 2. Depth distribution comparison
        ax2 = fig.add_subplot(gs[0, 1])
        if len(static_stats['depths']) > 0 and len(adaptive_stats['depths']) > 0:
            ax2.hist(adaptive_stats['depths'], bins=10, alpha=0.7,
                    label=f'Adaptive', color=tab10_colors[1], density=True)
            ax2.hist(static_stats['depths'], bins=10, alpha=0.7,
                    label=f'Static', color=tab10_colors[0], density=True)
            if len(new_detection_stats['depths']) > 0:
                ax2.hist(new_detection_stats['depths'], bins=10, alpha=0.7,
                        label=f'New only', color=tab10_colors[1], density=True,
                        histtype='step', linewidth=1.5)

        ax2.set_xlabel(r'Depth $d$ (0=close, 1=far)')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)

        # 3. Size vs Depth scatter plot
        ax3 = fig.add_subplot(gs[1, 0])
        if len(static_stats['sizes']) > 0:
            ax3.scatter(static_stats['depths'], np.log10(np.array(static_stats['sizes']) + 1),
                        alpha=0.5, s=8, color=tab10_colors[0], label='Static')

        if len(new_detection_stats['sizes']) > 0:
            ax3.scatter(new_detection_stats['depths'],
                        np.log10(np.array(new_detection_stats['sizes']) + 1),
                        alpha=0.5, s=8, color=tab10_colors[1], label='DCT')

        ax3.set_xlabel(r'Depth $d$ (0=close, 1=far)')
        ax3.set_ylabel(r'$\log_{10}$(Area + 1)')
        ax3.legend(labelspacing=0.0,columnspacing=0.3, markerscale=1.5, handlelength=1.3, handletextpad=0.2,)
        ax3.grid(True, alpha=0.3)

        # 4. Per-class detection counts
        ax4 = fig.add_subplot(gs[1, 1])

        static_class_counts = Counter(static_stats['classes'])
        adaptive_class_counts = Counter(adaptive_stats['classes'])
        new_class_counts = Counter(new_detection_stats['classes'])
        all_classes = sorted(set(static_class_counts.keys()) | set(adaptive_class_counts.keys()))
        x = np.arange(len(all_classes))
        new_counts = [new_class_counts.get(c, 0) for c in all_classes]
        ax4.bar(x, new_counts, label='New only', color=tab10_colors[1])
        ax4.set_ylabel('New Detections')
        ax4.set_xticks(x)
        ax4.set_xticklabels(all_classes, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(ADD_PATH + f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/size_depth_class_analysis.png', bbox_inches='tight')
        plt.savefig(ADD_PATH + f'/DepthPrior/object_detection/plots_DCT/{plot_addition}/size_depth_class_analysis.svg', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset processing script')
    
    parser.add_argument('--dataset', 
                       type=str, 
                       default='KITTI',
                       choices=['KITTI', 'MSCOCO', 'SUNRGBD', 'visdrone'],
                       help='Dataset to use (default: KITTI)')
    
    parser.add_argument('--yolo', 
                       action='store_true',
                       help='Enable YOLO mode (default: False)')
    
    parser.add_argument('--val-activate', 
                       action='store_true',
                       help='Activate validation (default: False)')
    
    args = parser.parse_args()
    
    # Use the arguments
    dataset = args.dataset
    yolo = args.yolo
    val_activate = args.val_activate
    run_function(dataset, yolo, val_activate)