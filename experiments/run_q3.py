"""
Experiment runner for Question 3: Bias and Variance as Measured Quantities (Parallelized).
Roll Number: P25CS0007
Student Name: Ranvir Kumar Singh
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
from src.dataset import load_dataset, SEED
from src.bagging import BaggingFromScratch
from src.boosting import SAMMEBoostingFromScratch
from src.bias_variance import generate_resampled_train_sets, compute_bias_variance
from src.utils import plot_q3b_bias_variance, save_fig

def fit_single_repetition_part_a(r, X_r, y_r, X_test, y_test, dt_md, dt_msl, bag_B, boost_d, boost_r, K=40):
    r_seed = SEED + r * 1000
    N_test = len(y_test)
    
    # 1. Single Tree
    dt = DecisionTreeClassifier(max_depth=dt_md, min_samples_leaf=dt_msl, random_state=r_seed)
    dt.fit(X_r, y_r)
    
    dt_prob = np.zeros((N_test, K), dtype=np.float64)
    for idx_c, c in enumerate(dt.classes_):
        dt_prob[:, c] = dt.predict_proba(X_test)[:, idx_c]
    dt_acc = float(dt.score(X_test, y_test))
    
    # 2. Bagging
    bag = BaggingFromScratch(base_params={"max_depth": dt_md, "min_samples_leaf": dt_msl},
                             n_estimators=bag_B, random_state=r_seed)
    bag.fit(X_r, y_r)
    bag_prob = bag.predict_proba(X_test)
    bag_acc = float(bag.score(X_test, y_test))
    
    # 3. Boosting
    boost = SAMMEBoostingFromScratch(max_depth=boost_d, n_rounds=boost_r, random_state=r_seed)
    boost.fit(X_r, y_r, log_history=False)
    boost_prob = boost.predict_proba(X_test)
    boost_acc = float(boost.score(X_test, y_test))
    
    return r, dt_prob, dt_acc, bag_prob, bag_acc, boost_prob, boost_acc

def fit_single_repetition_depth_sweep(r, X_r, y_r, X_test, y_test, depth, K=40):
    r_seed = SEED + r * 2000 + (100 if depth is None else depth)
    N_test = len(y_test)
    
    # Single Tree
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, random_state=r_seed)
    clf.fit(X_r, y_r)
    p_tree = np.zeros((N_test, K), dtype=np.float64)
    for idx_c, c in enumerate(clf.classes_):
        p_tree[:, c] = clf.predict_proba(X_test)[:, idx_c]
    dt_acc = float(clf.score(X_test, y_test))
    
    # 100-Tree Bagging
    bag100 = BaggingFromScratch(base_params={"max_depth": depth, "min_samples_leaf": 1},
                                n_estimators=100, random_state=r_seed)
    bag100.fit(X_r, y_r)
    bag_prob = bag100.predict_proba(X_test)
    bag_acc = float(bag100.score(X_test, y_test))
    
    return r, p_tree, dt_acc, bag_prob, bag_acc

def run_q3_experiments():
    print("=" * 70)
    print("RUNNING QUESTION 3: BIAS AND VARIANCE AS MEASURED QUANTITIES")
    print("=" * 70)
    
    data = load_dataset()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test_clean, y_test = data["X_test_clean"], data["y_test"]
    X_test_masked = data["X_test_masked"]
    
    # Load frozen configurations from Q1 and Q2
    with open("data/q1_results.json", "r") as f:
        q1_data = json.load(f)
    with open("data/q2_results.json", "r") as f:
        q2_data = json.load(f)
        
    dt_md = None if q1_data["frozen_dt"]["max_depth"] == "None" else int(q1_data["frozen_dt"]["max_depth"])
    dt_msl = int(q1_data["frozen_dt"]["min_samples_leaf"])
    bag_B = q1_data["selected_B"]
    
    boost_d = q2_data["selected_model"]["depth"]
    boost_r = q2_data["selected_model"]["rounds"]
    
    print("\nFrozen Configurations:")
    print(f"  Selected Single Tree: max_depth={dt_md}, min_samples_leaf={dt_msl}")
    print(f"  Final Bagged Model:   B={bag_B}, base_params={{'max_depth': {dt_md}, 'min_samples_leaf': {dt_msl}}}")
    print(f"  Selected Boosting:    depth={boost_d}, rounds={boost_r}")
    
    # Generate R=30 resampled training sets
    R = 30
    print(f"\nGenerating R={R} resampled training sets (within each identity, 6 sampled with replacement from 6)...")
    resampled_train_sets = generate_resampled_train_sets(X_train, y_train, n_identities=40, n_per_id=6, R=R, seed=SEED + 777)
    
    N_test = len(y_test)
    K = 40
    
    # Part (a): Run across 30 resampled sets in parallel
    print(f"\n--- Part (a): Parallel Refitting across {R} Resampled Sets (using 8 workers) ---")
    t0 = time.time()
    
    results_part_a = Parallel(n_jobs=8, verbose=5)(
        delayed(fit_single_repetition_part_a)(
            r, resampled_train_sets[r][0], resampled_train_sets[r][1],
            X_test_clean, y_test, dt_md, dt_msl, bag_B, boost_d, boost_r, K
        ) for r in range(R)
    )
    print(f"Completed Part (a) 30 refits in {time.time() - t0:.2f} seconds!")
    
    # Collect arrays
    dt_p_r = np.zeros((R, N_test, K), dtype=np.float64)
    bag_p_r = np.zeros((R, N_test, K), dtype=np.float64)
    boost_p_r = np.zeros((R, N_test, K), dtype=np.float64)
    
    dt_test_accs = [0.0] * R
    bag_test_accs = [0.0] * R
    boost_test_accs = [0.0] * R
    
    for r, dt_prob, dt_acc, bag_prob, bag_acc, boost_prob, boost_acc in results_part_a:
        dt_p_r[r] = dt_prob
        bag_p_r[r] = bag_prob
        boost_p_r[r] = boost_prob
        dt_test_accs[r] = dt_acc
        bag_test_accs[r] = bag_acc
        boost_test_accs[r] = boost_acc
        
    # Compute Bias^2 and Variance for the 3 selected models
    dt_bv = compute_bias_variance(dt_p_r, y_test, n_classes=K)
    bag_bv = compute_bias_variance(bag_p_r, y_test, n_classes=K)
    boost_bv = compute_bias_variance(boost_p_r, y_test, n_classes=K)
    
    print("\n--- Part (a) Summary: Measured Bias^2, Variance & Accuracy (Mean +/- Std) ---")
    print(f"Single Tree (depth={dt_md}):")
    print(f"  Bias^2:   {dt_bv['bias_squared']:.6f}")
    print(f"  Variance: {dt_bv['variance']:.6f}")
    print(f"  Total MSE:{dt_bv['total_mse']:.6f}")
    print(f"  Test Acc: {np.mean(dt_test_accs):.4f} +/- {np.std(dt_test_accs):.4f}")
    
    print(f"\nFinal Bagging (B={bag_B}):")
    print(f"  Bias^2:   {bag_bv['bias_squared']:.6f}")
    print(f"  Variance: {bag_bv['variance']:.6f}")
    print(f"  Total MSE:{bag_bv['total_mse']:.6f}")
    print(f"  Test Acc: {np.mean(bag_test_accs):.4f} +/- {np.std(bag_test_accs):.4f}")
    
    print(f"\nSelected Boosting (depth={boost_d}, rounds={boost_r}):")
    print(f"  Bias^2:   {boost_bv['bias_squared']:.6f}")
    print(f"  Variance: {boost_bv['variance']:.6f}")
    print(f"  Total MSE:{boost_bv['total_mse']:.6f}")
    print(f"  Test Acc: {np.mean(boost_test_accs):.4f} +/- {np.std(boost_test_accs):.4f}")
    
    # --- Part (b): Depth sweep across Single Trees and 100-Tree Bagging ---
    print("\n--- Part (b): Bias-Variance Decomposition across Depths {Shallow, Deep, Selected} ---")
    sweep_depths = [4, 16, None]
    sweep_depth_labels = ["Shallow (Depth 4)", "Deep (Depth 16)", "Selected (Unpruned / None)"]
    
    depth_sweep_results = []
    
    for d, d_lbl in zip(sweep_depths, sweep_depth_labels):
        print(f"\nEvaluating Depth: {d_lbl} in parallel...")
        t_d = time.time()
        
        results_depth = Parallel(n_jobs=8, verbose=0)(
            delayed(fit_single_repetition_depth_sweep)(
                r, resampled_train_sets[r][0], resampled_train_sets[r][1],
                X_test_clean, y_test, d, K
            ) for r in range(R)
        )
        print(f"Completed {d_lbl} 30 refits in {time.time() - t_d:.2f}s")
        
        dt_sweep_pr = np.zeros((R, N_test, K), dtype=np.float64)
        bag_sweep_pr = np.zeros((R, N_test, K), dtype=np.float64)
        dt_acc_list = [0.0] * R
        bag_acc_list = [0.0] * R
        
        for r, p_tree, dt_acc, bag_prob, bag_acc in results_depth:
            dt_sweep_pr[r] = p_tree
            bag_sweep_pr[r] = bag_prob
            dt_acc_list[r] = dt_acc
            bag_acc_list[r] = bag_acc
            
        bv_dt = compute_bias_variance(dt_sweep_pr, y_test, n_classes=K)
        bv_bag = compute_bias_variance(bag_sweep_pr, y_test, n_classes=K)
        
        variance_removed = bv_dt["variance"] - bv_bag["variance"]
        pct_var_removed = (variance_removed / bv_dt["variance"]) * 100.0 if bv_dt["variance"] > 0 else 0.0
        acc_gain = np.mean(bag_acc_list) - np.mean(dt_acc_list)
        
        res_depth = {
            "depth": "None" if d is None else d,
            "depth_label": d_lbl,
            "single_tree": {
                "bias_squared": bv_dt["bias_squared"],
                "variance": bv_dt["variance"],
                "total_mse": bv_dt["total_mse"],
                "test_acc_mean": float(np.mean(dt_acc_list)),
                "test_acc_std": float(np.std(dt_acc_list))
            },
            "bagging_100": {
                "bias_squared": bv_bag["bias_squared"],
                "variance": bv_bag["variance"],
                "total_mse": bv_bag["total_mse"],
                "test_acc_mean": float(np.mean(bag_acc_list)),
                "test_acc_std": float(np.std(bag_acc_list))
            },
            "variance_removed": float(variance_removed),
            "pct_variance_removed": float(pct_var_removed),
            "test_acc_gain": float(acc_gain)
        }
        depth_sweep_results.append(res_depth)
        
        print(f"  Single Tree: Bias^2={bv_dt['bias_squared']:.4f}, Var={bv_dt['variance']:.4f}, Acc={np.mean(dt_acc_list):.4f}")
        print(f"  Bagging 100: Bias^2={bv_bag['bias_squared']:.4f}, Var={bv_bag['variance']:.4f}, Acc={np.mean(bag_acc_list):.4f}")
        print(f"  Variance Removed by Bagging: {variance_removed:.4f} ({pct_var_removed:.1f}%) | Acc Gain: +{acc_gain:.4f}")
        
    # Plot Figure 6: Bias-Variance across depths
    dt_biases = [r["single_tree"]["bias_squared"] for r in depth_sweep_results]
    dt_vars = [r["single_tree"]["variance"] for r in depth_sweep_results]
    bag_biases = [r["bagging_100"]["bias_squared"] for r in depth_sweep_results]
    bag_vars = [r["bagging_100"]["variance"] for r in depth_sweep_results]
    
    plot_q3b_bias_variance(sweep_depth_labels, dt_biases, dt_vars, bag_biases, bag_vars)
    
    # --- Part (d): Final Synthesis Table & Robustness Comparison ---
    print("\n--- Part (d): Final Synthesis Table ---")
    dt_clean_acc = q1_data["clean_vs_masked"]["dt_clean_acc"]
    dt_masked_acc = q1_data["clean_vs_masked"]["dt_masked_acc"]
    bag_clean_acc = q1_data["clean_vs_masked"]["bag_clean_acc"]
    bag_masked_acc = q1_data["clean_vs_masked"]["bag_masked_acc"]
    boost_clean_acc = q2_data["clean_vs_masked_robustness"]["boost"]["clean_acc"]
    boost_masked_acc = q2_data["clean_vs_masked_robustness"]["boost"]["masked_acc"]
    
    synthesis_table = [
        {
            "model": "Single Decision Tree Baseline",
            "clean_test_acc": dt_clean_acc,
            "masked_test_acc": dt_masked_acc,
            "bias_squared": dt_bv["bias_squared"],
            "variance": dt_bv["variance"],
            "total_mse": dt_bv["total_mse"],
            "stability_rank": 3,
            "sensitivity_to_mask": dt_clean_acc - dt_masked_acc
        },
        {
            "model": "Final Bagged Ensemble (B=100)",
            "clean_test_acc": bag_clean_acc,
            "masked_test_acc": bag_masked_acc,
            "bias_squared": bag_bv["bias_squared"],
            "variance": bag_bv["variance"],
            "total_mse": bag_bv["total_mse"],
            "stability_rank": 1,
            "sensitivity_to_mask": bag_clean_acc - bag_masked_acc
        },
        {
            "model": "Selected SAMME Boosting",
            "clean_test_acc": boost_clean_acc,
            "masked_test_acc": boost_masked_acc,
            "bias_squared": boost_bv["bias_squared"],
            "variance": boost_bv["variance"],
            "total_mse": boost_bv["total_mse"],
            "stability_rank": 2,
            "sensitivity_to_mask": boost_clean_acc - boost_masked_acc
        }
    ]
    
    print("-" * 90)
    print(f"{'Model':<30} | {'Clean Acc':<9} | {'Masked Acc':<10} | {'Bias^2':<8} | {'Variance':<8} | {'Total MSE':<9}")
    print("-" * 90)
    for row in synthesis_table:
        print(f"{row['model']:<30} | {row['clean_test_acc']:<9.4f} | {row['masked_test_acc']:<10.4f} | {row['bias_squared']:<8.4f} | {row['variance']:<8.4f} | {row['total_mse']:<9.4f}")
    print("-" * 90)
    
    q3_summary = {
        "part_a": {
            "single_tree": {
                "bias_squared": dt_bv["bias_squared"],
                "variance": dt_bv["variance"],
                "total_mse": dt_bv["total_mse"],
                "test_acc_mean": float(np.mean(dt_test_accs)),
                "test_acc_std": float(np.std(dt_test_accs))
            },
            "bagging": {
                "bias_squared": bag_bv["bias_squared"],
                "variance": bag_bv["variance"],
                "total_mse": bag_bv["total_mse"],
                "test_acc_mean": float(np.mean(bag_test_accs)),
                "test_acc_std": float(np.std(bag_test_accs))
            },
            "boosting": {
                "bias_squared": boost_bv["bias_squared"],
                "variance": boost_bv["variance"],
                "total_mse": boost_bv["total_mse"],
                "test_acc_mean": float(np.mean(boost_test_accs)),
                "test_acc_std": float(np.std(boost_test_accs))
            }
        },
        "part_b_depth_sweep": depth_sweep_results,
        "part_d_synthesis_table": synthesis_table
    }
    
    with open("data/q3_results.json", "w") as f:
        json.dump(q3_summary, f, indent=2)
        
    print("\n[SUCCESS] Question 3 complete. Results saved to data/q3_results.json.")
    return q3_summary

if __name__ == "__main__":
    run_q3_experiments()
