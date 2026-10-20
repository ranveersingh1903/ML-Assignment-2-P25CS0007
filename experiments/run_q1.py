"""
Experiment runner for Question 1: Decision Tree Baseline & Bagging from Scratch.
Roll Number: P25CS0007
Student Name: Ranvir Kumar Singh
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import load_dataset, SEED
from src.decision_tree import run_decision_tree_grid
from src.bagging import BaggingFromScratch
from src.utils import plot_q1c_diagnostics, save_fig

def run_q1_experiments():
    print("=" * 70)
    print("RUNNING QUESTION 1: DECISION TREE BASELINE & BAGGING FROM SCRATCH")
    print("=" * 70)
    
    data = load_dataset()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test_clean, y_test = data["X_test_clean"], data["y_test"]
    X_test_masked = data["X_test_masked"]
    
    # --- Part (a): Decision Tree Baseline ---
    print("\n--- Part (a): Decision Tree Grid Search ---")
    df_dt, best_dt_setting = run_decision_tree_grid()
    
    frozen_params = {
        "max_depth": None if best_dt_setting["max_depth"] == "None" else int(best_dt_setting["max_depth"]),
        "min_samples_leaf": int(best_dt_setting["min_samples_leaf"])
    }
    frozen_tree = best_dt_setting["model"]
    
    # --- Part (b) & (c): Bagging Sweep over B ---
    print("\n--- Part (b) & (c): Bagging Sweep over B in {1, 5, 10, 25, 50, 100, 200} ---")
    B_list = [1, 5, 10, 25, 50, 100, 200]
    
    # Theoretical expectation for N=240
    N_train = len(y_train)
    theo_distinct = N_train * (1.0 - ((N_train - 1.0) / N_train) ** N_train)
    print(f"Theoretical expected distinct samples in N={N_train} bootstrap: {theo_distinct:.4f} ({theo_distinct/N_train*100:.2f}%)")
    
    b_results = []
    trained_ensembles = {}
    
    for B in B_list:
        bag = BaggingFromScratch(base_params=frozen_params, n_estimators=B, random_state=SEED + B)
        bag.fit(X_train, y_train)
        trained_ensembles[B] = bag
        
        train_acc = bag.score(X_train, y_train)
        val_acc = bag.score(X_val, y_val)
        test_acc = bag.score(X_test_clean, y_test)
        
        avg_distinct, _ = bag.get_average_distinct_samples()
        oob_acc, n_oob = bag.compute_oob_accuracy(X_train, y_train)
        pairwise_disagree = bag.compute_pairwise_disagreement(X_val)
        mean_margin, _ = bag.compute_mean_vote_margin(X_val)
        
        res = {
            "B": B,
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "avg_distinct": float(avg_distinct),
            "oob_acc": float(oob_acc),
            "pairwise_disagreement": float(pairwise_disagree),
            "mean_vote_margin": float(mean_margin)
        }
        b_results.append(res)
        print(f"B={B:3d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | "
              f"Distinct: {avg_distinct:.2f} | OOB Acc: {oob_acc:.4f} | Disagree: {pairwise_disagree:.4f} | Margin: {mean_margin:.4f}")
              
    # Select final ensemble size using validation accuracy only
    # In case of tie, choose the more computationally parsimonious or stabilized B
    val_accs = [r["val_acc"] for r in b_results]
    max_val = max(val_accs)
    # Filter candidates with max_val
    best_B_candidates = [r["B"] for r in b_results if r["val_acc"] == max_val]
    # Standard choice: B=100 provides excellent stability and peak validation performance
    selected_B = 100 if 100 in best_B_candidates else best_B_candidates[0]
    print(f"\nSelected Final Bagged Ensemble Size (using val acc): B = {selected_B}")
    
    final_bag = trained_ensembles[selected_B]
    
    # Generate Figure 1: Q1(c) Diagnostics Plot
    oob_plot_vals = [r["oob_acc"] for r in b_results]
    disagree_plot_vals = [r["pairwise_disagreement"] for r in b_results]
    margin_plot_vals = [r["mean_vote_margin"] for r in b_results]
    plot_q1c_diagnostics(B_list, oob_plot_vals, disagree_plot_vals, margin_plot_vals, selected_B=selected_B)
    
    # --- Part (d): Comparison on Clean vs Masked Test Sets ---
    print("\n--- Part (d): Comparison of Single Tree vs. Final Bagged Model ---")
    
    # Single Tree evaluations
    dt_clean_acc = float(frozen_tree.score(X_test_clean, y_test))
    dt_masked_acc = float(frozen_tree.score(X_test_masked, y_test))
    
    # Single tree prediction confidence (max proba)
    dt_clean_proba = frozen_tree.predict_proba(X_test_clean)
    dt_masked_proba = frozen_tree.predict_proba(X_test_masked)
    dt_clean_conf = float(np.mean(np.max(dt_clean_proba, axis=1)))
    dt_masked_conf = float(np.mean(np.max(dt_masked_proba, axis=1)))
    
    # Bagging evaluations
    bag_clean_acc = float(final_bag.score(X_test_clean, y_test))
    bag_masked_acc = float(final_bag.score(X_test_masked, y_test))
    
    bag_clean_conf_vec = final_bag.get_prediction_confidence(X_test_clean)
    bag_masked_conf_vec = final_bag.get_prediction_confidence(X_test_masked)
    bag_clean_conf = float(np.mean(bag_clean_conf_vec))
    bag_masked_conf = float(np.mean(bag_masked_conf_vec))
    
    print(f"Single Tree   | Clean Acc: {dt_clean_acc:.4f} (Conf: {dt_clean_conf:.4f}) | Masked Acc: {dt_masked_acc:.4f} (Conf: {dt_masked_conf:.4f})")
    print(f"Final Bagging | Clean Acc: {bag_clean_acc:.4f} (Conf: {bag_clean_conf:.4f}) | Masked Acc: {bag_masked_acc:.4f} (Conf: {bag_masked_conf:.4f})")
    
    # Identify cases:
    # 1. Two faces where bagging corrects the tree (Single Tree Wrong, Bagging Correct on clean test)
    # 2. Two faces where both are wrong on clean test
    dt_preds = frozen_tree.predict(X_test_clean)
    bag_preds = final_bag.predict(X_test_clean)
    
    corrected_indices = np.where((dt_preds != y_test) & (bag_preds == y_test))[0]
    both_wrong_indices = np.where((dt_preds != y_test) & (bag_preds != y_test))[0]
    
    print(f"\nNumber of faces corrected by bagging: {len(corrected_indices)}")
    print(f"Number of faces where both fail: {len(both_wrong_indices)}")
    
    chosen_corrected = corrected_indices[:2].tolist()
    chosen_both_wrong = both_wrong_indices[:2].tolist()
    
    print(f"Chosen corrected face indices (in test set): {chosen_corrected}")
    print(f"Chosen both wrong face indices (in test set): {chosen_both_wrong}")
    
    # Plot Figure 2: The 4 faces and their vote distributions
    fig, axes = plt.subplots(4, 2, figsize=(8.0, 9.5), gridspec_kw={"width_ratios": [1, 2.5]})
    selected_cases = [
        ("Bagging Corrects Tree (Case 1)", chosen_corrected[0]),
        ("Bagging Corrects Tree (Case 2)", chosen_corrected[1]),
        ("Both Models Wrong (Case 1)", chosen_both_wrong[0]),
        ("Both Models Wrong (Case 2)", chosen_both_wrong[1])
    ]
    
    bag_proba_all = final_bag.predict_proba(X_test_clean)
    
    face_details = []
    
    for row, (label, idx) in enumerate(selected_cases):
        face_img = X_test_clean[idx].reshape(64, 64)
        true_lbl = int(y_test[idx])
        dt_pred = int(dt_preds[idx])
        bag_pred = int(bag_preds[idx])
        votes = bag_proba_all[idx] * selected_B # integer vote counts
        
        # Display Face
        ax_img = axes[row, 0]
        ax_img.imshow(face_img, cmap="gray")
        ax_img.axis("off")
        ax_img.set_title(f"Face #{idx} (True: {true_lbl})\nTree: {dt_pred} | Bag: {bag_pred}", fontsize=9)
        
        # Display Top 5 Vote Counts
        ax_bar = axes[row, 1]
        top5_idx = np.argsort(votes)[-5:][::-1]
        top5_votes = votes[top5_idx]
        
        bar_colors = []
        for c in top5_idx:
            if c == true_lbl:
                bar_colors.append("#2ca02c") # Green for ground truth
            elif c == dt_pred:
                bar_colors.append("#d62728") # Red for single tree
            else:
                bar_colors.append("#1f77b4") # Blue
                
        bars = ax_bar.bar([f"Class {c}" for c in top5_idx], top5_votes, color=bar_colors, width=0.55)
        ax_bar.set_ylabel("Vote Count (B=100)")
        ax_bar.set_title(f"{label} [Green=Ground Truth, Red=Tree Choice]", fontsize=9)
        ax_bar.set_ylim(0, max(top5_votes) * 1.25 + 5)
        
        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax_bar.annotate(f"{int(height)}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
                            
        face_details.append({
            "case_type": label,
            "test_index": idx,
            "true_identity": true_lbl,
            "single_tree_pred": dt_pred,
            "bagging_pred": bag_pred,
            "top5_classes": top5_idx.tolist(),
            "top5_votes": top5_votes.tolist()
        })
        
    plt.tight_layout()
    fig2_path = save_fig(fig, "fig2_q1d_face_analysis.png")
    
    # Save all Q1 data to JSON for paper writing
    q1_summary = {
        "frozen_dt": {
            "max_depth": str(frozen_params["max_depth"]),
            "min_samples_leaf": frozen_params["min_samples_leaf"],
            "train_acc": best_dt_setting["train_acc"],
            "val_acc": best_dt_setting["val_acc"],
            "test_acc": best_dt_setting["test_acc"],
            "n_leaves": best_dt_setting["n_leaves"],
            "mean_samples_per_leaf": best_dt_setting["mean_samples_per_leaf"],
            "singleton_fraction": best_dt_setting["singleton_fraction"]
        },
        "dt_grid_table": df_dt.drop(columns=["model"]).to_dict(orient="records"),
        "bagging_sweep": b_results,
        "theoretical_distinct": theo_distinct,
        "selected_B": selected_B,
        "clean_vs_masked": {
            "dt_clean_acc": dt_clean_acc,
            "dt_clean_conf": dt_clean_conf,
            "dt_masked_acc": dt_masked_acc,
            "dt_masked_conf": dt_masked_conf,
            "bag_clean_acc": bag_clean_acc,
            "bag_clean_conf": bag_clean_conf,
            "bag_masked_acc": bag_masked_acc,
            "bag_masked_conf": bag_masked_conf
        },
        "inspected_faces": face_details
    }
    
    with open("data/q1_results.json", "w") as f:
        json.dump(q1_summary, f, indent=2)
        
    print("\n[SUCCESS] Question 1 complete. Results saved to data/q1_results.json.")
    return q1_summary

if __name__ == "__main__":
    run_q1_experiments()
