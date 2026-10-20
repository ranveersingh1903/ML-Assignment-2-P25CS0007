"""
Experiment runner for Question 2: Multiclass Boosting from Scratch (SAMME).
Roll Number: P25CS0007
Student Name: Ranvir Kumar Singh
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import load_dataset, SEED
from src.boosting import SAMMEBoostingFromScratch, run_samme_toy_example
from src.bagging import BaggingFromScratch
from src.decision_tree import run_decision_tree_grid
from src.utils import plot_q2b_boosting_dynamics, save_fig

def run_q2_experiments():
    print("=" * 70)
    print("RUNNING QUESTION 2: MULTICLASS BOOSTING FROM SCRATCH (SAMME)")
    print("=" * 70)
    
    data = load_dataset()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test_clean, y_test = data["X_test_clean"], data["y_test"]
    X_test_masked = data["X_test_masked"]
    
    # --- Part (a): SAMME Toy Example Verification ---
    print("\n--- Part (a): SAMME Toy 3-Class Example Verification ---")
    toy_log = run_samme_toy_example()
    
    # --- Part (b): Hyperparameter Search on Olivetti Faces ---
    print("\n--- Part (b): Grid Search over Weak Learner Depths {1, 2, 3, 4} and Rounds {10, 25, 50, 100} ---")
    depths = [1, 2, 3, 4]
    rounds_list = [10, 25, 50, 100]
    
    tuning_results = []
    models_dict = {}
    
    for d in depths:
        for r in rounds_list:
            t0 = time.time()
            clf = SAMMEBoostingFromScratch(max_depth=d, n_rounds=r, random_state=SEED + d * 100 + r)
            clf.fit(X_train, y_train, log_history=True)
            fit_time = time.time() - t0
            
            train_acc = clf.score(X_train, y_train)
            val_acc = clf.score(X_val, y_val)
            test_acc = clf.score(X_test_clean, y_test)
            
            res = {
                "depth": d,
                "rounds": r,
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "fit_time": float(fit_time),
                "n_actual_estimators": len(clf.estimators_),
                "stopped_early": clf.stopped_early_
            }
            tuning_results.append(res)
            models_dict[(d, r)] = clf
            print(f"Depth: {d:2d} | Rounds: {r:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Fit Time: {fit_time:.2f}s | Est: {len(clf.estimators_)}")
            
    # Select best model using clean validation accuracy only
    tuning_results_sorted = sorted(tuning_results, key=lambda x: (x["val_acc"], -x["depth"], -x["rounds"]), reverse=True)
    best_config = tuning_results_sorted[0]
    sel_d = best_config["depth"]
    sel_r = best_config["rounds"]
    selected_boosting = models_dict[(sel_d, sel_r)]
    
    print("\n--- Selected Frozen Boosting Configuration (Val Acc Only) ---")
    print(f"Weak Learner Depth: {sel_d}, Rounds: {sel_r}")
    print(f"Train Acc: {best_config['train_acc']:.4f}, Val Acc: {best_config['val_acc']:.4f}, Test Acc: {best_config['test_acc']:.4f}")
    
    # Plotting Figure 3: Training dynamics over rounds for selected model
    # Curves:
    # 1. Weighted training error eps_t
    # 2. Learner weight alpha_t
    # 3. Maximum sample weight max_i w_i^(t)
    # 4. Total weight carried by 10 heaviest training images
    w_errors = selected_boosting.weighted_errors_
    alphas = selected_boosting.alphas_
    
    # Weight history has shape (rounds, N) or (rounds+1, N)
    weight_hist = np.array(selected_boosting.sample_weight_history_[:len(w_errors)]) # (T, N)
    max_weights = np.max(weight_hist, axis=1)
    
    # Sort weights along samples axis and sum top 10
    sorted_weights = np.sort(weight_hist, axis=1) # ascending
    top10_weights = np.sum(sorted_weights[:, -10:], axis=1)
    
    plot_q2b_boosting_dynamics(w_errors, alphas, max_weights, top10_weights)
    
    # --- Part (c): Comparison of Boosting vs Bagging (25, 50, 100 learners) ---
    print("\n--- Part (c): Comparison of Boosting vs. Bagging across {25, 50, 100} Learners ---")
    comp_rounds = [25, 50, 100]
    comp_results = []
    
    # Frozen decision tree params for bagging
    frozen_bag_params = {"max_depth": None, "min_samples_leaf": 1}
    
    for M in comp_rounds:
        # 1. Boosting with depth = sel_d
        t0 = time.time()
        boost_m = SAMMEBoostingFromScratch(max_depth=sel_d, n_rounds=M, random_state=SEED + M)
        boost_m.fit(X_train, y_train, log_history=False)
        boost_time = time.time() - t0
        
        b_train = boost_m.score(X_train, y_train)
        b_val = boost_m.score(X_val, y_val)
        b_test = boost_m.score(X_test_clean, y_test)
        
        # 2. Bagging with M estimators
        t0 = time.time()
        bag_m = BaggingFromScratch(base_params=frozen_bag_params, n_estimators=M, random_state=SEED + M)
        bag_m.fit(X_train, y_train)
        bag_time = time.time() - t0
        
        bag_train = bag_m.score(X_train, y_train)
        bag_val = bag_m.score(X_val, y_val)
        bag_test = bag_m.score(X_test_clean, y_test)
        
        comp_res = {
            "learners": M,
            "boosting": {
                "train_acc": float(b_train),
                "val_acc": float(b_val),
                "test_acc": float(b_test),
                "time_sec": float(boost_time)
            },
            "bagging": {
                "train_acc": float(bag_train),
                "val_acc": float(bag_val),
                "test_acc": float(bag_test),
                "time_sec": float(bag_time)
            }
        }
        comp_results.append(comp_res)
        print(f"M={M:3d} Learners:")
        print(f"  Boosting | Train: {b_train:.4f} | Val: {b_val:.4f} | Test: {b_test:.4f} | Time: {boost_time:.2f}s")
        print(f"  Bagging  | Train: {bag_train:.4f} | Val: {bag_val:.4f} | Test: {bag_test:.4f} | Time: {bag_time:.2f}s")
        
    # Inspect one training face that receives a very large boosting weight
    # Look at final weights across all training images
    final_weights = selected_boosting.sample_weight_history_[-1]
    heaviest_idx = int(np.argmax(final_weights))
    heaviest_weight = float(final_weights[heaviest_idx])
    heaviest_face_img = X_train[heaviest_idx].reshape(64, 64)
    heaviest_true_class = int(y_train[heaviest_idx])
    
    # Prediction history across rounds for this sample
    heavy_sample_x = X_train[heaviest_idx:heaviest_idx+1]
    preds_history = [int(clf.predict(heavy_sample_x)[0]) for clf in selected_boosting.estimators_]
    correct_history = [int(p == heaviest_true_class) for p in preds_history]
    sample_weight_history = [float(selected_boosting.sample_weight_history_[t][heaviest_idx]) for t in range(len(selected_boosting.estimators_))]
    
    print(f"\nHeaviest Training Sample Index: {heaviest_idx} (True Class: {heaviest_true_class})")
    print(f"Initial Weight: {1.0/len(y_train):.6f} | Final Weight: {heaviest_weight:.6f} ({heaviest_weight / (1.0/len(y_train)):.2f}x average)")
    print(f"Fraction of rounds misclassified by weak learners: {1.0 - np.mean(correct_history):.2%}")
    
    # Plot Figure 4: Heaviest training face and its trajectory
    fig, (ax_img, ax_traj) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1, 2.2]})
    ax_img.imshow(heaviest_face_img, cmap="gray")
    ax_img.axis("off")
    ax_img.set_title(f"Heaviest Train Face #{heaviest_idx}\nClass {heaviest_true_class} (W={heaviest_weight:.4f})")
    
    rounds_axis = np.arange(1, len(sample_weight_history) + 1)
    ax_traj.plot(rounds_axis, sample_weight_history, color="#d62728", marker="o", markersize=3, label="Sample Weight $w_i^{(t)}$")
    ax_traj.axhline(y=1.0/len(y_train), color="gray", linestyle=":", label="Uniform Weight (0.00417)")
    ax_traj.set_title("Sample Weight Evolution over Boosting Rounds")
    ax_traj.set_xlabel("Boosting Round ($t$)")
    ax_traj.set_ylabel("Normalized Weight")
    ax_traj.legend(loc="upper left")
    
    save_fig(fig, "fig4_q2c_heavy_sample.png")
    
    # --- Part (d): Robustness on Clean vs Masked Test Sets ---
    print("\n--- Part (d): Evaluation of Single Tree, Bagging, and Boosting on Clean vs Masked Test Sets ---")
    # Load frozen single tree and final bagging
    with open("data/q1_results.json", "r") as f:
        q1_data = json.load(f)
        
    frozen_tree_setting = q1_data["frozen_dt"]
    selected_B = q1_data["selected_B"]
    
    # Refit baseline single tree
    dt_md = None if frozen_tree_setting["max_depth"] == "None" else int(frozen_tree_setting["max_depth"])
    dt_msl = int(frozen_tree_setting["min_samples_leaf"])
    single_tree = run_decision_tree_grid()[1]["model"]
    
    # Refit final bagged model
    bag_model = BaggingFromScratch(base_params={"max_depth": dt_md, "min_samples_leaf": dt_msl},
                                   n_estimators=selected_B, random_state=SEED + selected_B)
    bag_model.fit(X_train, y_train)
    
    # Clean evaluations
    dt_clean_acc = float(single_tree.score(X_test_clean, y_test))
    bag_clean_acc = float(bag_model.score(X_test_clean, y_test))
    boost_clean_acc = float(selected_boosting.score(X_test_clean, y_test))
    
    dt_clean_conf = float(np.mean(np.max(single_tree.predict_proba(X_test_clean), axis=1)))
    bag_clean_conf = float(np.mean(bag_model.get_prediction_confidence(X_test_clean)))
    boost_clean_conf = float(np.mean(selected_boosting.get_prediction_confidence(X_test_clean)))
    
    # Masked evaluations
    dt_masked_acc = float(single_tree.score(X_test_masked, y_test))
    bag_masked_acc = float(bag_model.score(X_test_masked, y_test))
    boost_masked_acc = float(selected_boosting.score(X_test_masked, y_test))
    
    dt_masked_conf = float(np.mean(np.max(single_tree.predict_proba(X_test_masked), axis=1)))
    bag_masked_conf = float(np.mean(bag_model.get_prediction_confidence(X_test_masked)))
    boost_masked_conf = float(np.mean(selected_boosting.get_prediction_confidence(X_test_masked)))
    
    print("Model Comparison Summary:")
    print(f"Single Tree Baseline: Clean Acc={dt_clean_acc:.4f} (Conf={dt_clean_conf:.4f}) | Masked Acc={dt_masked_acc:.4f} (Conf={dt_masked_conf:.4f})")
    print(f"Final Bagging (B={selected_B}): Clean Acc={bag_clean_acc:.4f} (Conf={bag_clean_conf:.4f}) | Masked Acc={bag_masked_acc:.4f} (Conf={bag_masked_conf:.4f})")
    print(f"Selected Boosting:    Clean Acc={boost_clean_acc:.4f} (Conf={boost_clean_conf:.4f}) | Masked Acc={boost_masked_acc:.4f} (Conf={boost_masked_conf:.4f})")
    
    # Find one face for which bagging and boosting respond differently to the missing region
    bag_masked_preds = bag_model.predict(X_test_masked)
    boost_masked_preds = selected_boosting.predict(X_test_masked)
    
    # Case where Bagging is CORRECT on masked face, but Boosting is WRONG (or vice versa)
    bag_correct_boost_wrong = np.where((bag_masked_preds == y_test) & (boost_masked_preds != y_test))[0]
    boost_correct_bag_wrong = np.where((boost_masked_preds == y_test) & (bag_masked_preds != y_test))[0]
    
    print(f"Faces where Bagging is Correct and Boosting is Wrong on Masked test: {bag_correct_boost_wrong.tolist()}")
    print(f"Faces where Boosting is Correct and Bagging is Wrong on Masked test: {boost_correct_bag_wrong.tolist()}")
    
    if len(bag_correct_boost_wrong) > 0:
        diff_idx = int(bag_correct_boost_wrong[0])
        diff_type = "Bagging Correct, Boosting Wrong on Masked"
    else:
        diff_idx = int(boost_correct_bag_wrong[0])
        diff_type = "Boosting Correct, Bagging Wrong on Masked"
        
    true_id = int(y_test[diff_idx])
    b_pred = int(bag_masked_preds[diff_idx])
    bst_pred = int(boost_masked_preds[diff_idx])
    
    # Plot Figure 5: Differential robustness face
    fig, (ax_clean, ax_mask, ax_vote) = plt.subplots(1, 3, figsize=(9.0, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.8]})
    ax_clean.imshow(X_test_clean[diff_idx].reshape(64, 64), cmap="gray")
    ax_clean.axis("off")
    ax_clean.set_title(f"Clean Face #{diff_idx}\n(True Class: {true_id})", fontsize=9)
    
    ax_mask.imshow(X_test_masked[diff_idx].reshape(64, 64), cmap="gray")
    ax_mask.axis("off")
    ax_mask.set_title(f"Masked Face (16x16)\nBag: {b_pred} | Boost: {bst_pred}", fontsize=9)
    
    # Votes comparison
    bag_proba = bag_model.predict_proba(X_test_masked[diff_idx:diff_idx+1])[0]
    boost_proba = selected_boosting.predict_proba(X_test_masked[diff_idx:diff_idx+1])[0]
    
    # Top classes by max proba across either
    top_c = np.unique(np.concatenate([np.argsort(bag_proba)[-3:], np.argsort(boost_proba)[-3:]]))
    x_axis = np.arange(len(top_c))
    width = 0.35
    
    ax_vote.bar(x_axis - width/2, bag_proba[top_c], width, label="Bagging Vote Frac", color="#2ca02c")
    ax_vote.bar(x_axis + width/2, boost_proba[top_c], width, label="Boosting Norm Vote", color="#ff7f0e")
    ax_vote.set_xticks(x_axis)
    ax_vote.set_xticklabels([f"Class {c}" + ("*" if c == true_id else "") for c in top_c])
    ax_vote.set_ylabel("Vote / Probability")
    ax_vote.set_title("Confidence on Masked Face (*=Ground Truth)", fontsize=9)
    ax_vote.legend(loc="upper right", fontsize=8)
    
    save_fig(fig, "fig5_q2d_masked_diff_face.png")
    
    q2_summary = {
        "toy_verification": {
            "initial_weights": toy_log["initial_weights"].tolist(),
            "predictions": toy_log["predictions"].tolist(),
            "weighted_error": toy_log["weighted_error"],
            "alpha": toy_log["alpha"],
            "updated_weights": toy_log["updated_weights"].tolist()
        },
        "grid_search": tuning_results,
        "selected_model": {
            "depth": sel_d,
            "rounds": sel_r,
            "train_acc": best_config["train_acc"],
            "val_acc": best_config["val_acc"],
            "test_acc": best_config["test_acc"],
            "fit_time": best_config["fit_time"]
        },
        "comparison_rounds": comp_results,
        "heaviest_sample": {
            "train_index": heaviest_idx,
            "true_class": heaviest_true_class,
            "final_weight": heaviest_weight,
            "misclassification_rate_in_rounds": float(1.0 - np.mean(correct_history))
        },
        "clean_vs_masked_robustness": {
            "dt": {"clean_acc": dt_clean_acc, "clean_conf": dt_clean_conf, "masked_acc": dt_masked_acc, "masked_conf": dt_masked_conf},
            "bag": {"clean_acc": bag_clean_acc, "clean_conf": bag_clean_conf, "masked_acc": bag_masked_acc, "masked_conf": bag_masked_conf},
            "boost": {"clean_acc": boost_clean_acc, "clean_conf": boost_clean_conf, "masked_acc": boost_masked_acc, "masked_conf": boost_masked_conf}
        },
        "differential_face": {
            "test_index": diff_idx,
            "case_type": diff_type,
            "true_class": true_id,
            "bagging_pred": b_pred,
            "boosting_pred": bst_pred
        }
    }
    
    with open("data/q2_results.json", "w") as f:
        json.dump(q2_summary, f, indent=2)
        
    print("\n[SUCCESS] Question 2 complete. Results saved to data/q2_results.json.")
    return q2_summary

if __name__ == "__main__":
    run_q2_experiments()
