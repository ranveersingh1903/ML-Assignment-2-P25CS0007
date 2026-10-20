"""
Plotting and utility functions for CVPR publication-quality figures and logs.
Roll Number: P25CS0007
Student Name: Ranvir Kumar Singh
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib publication defaults
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "font.family": "sans-serif",
    "figure.autolayout": True,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

def save_fig(fig, filename, fig_dir="figures", dpi=300):
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return out_path

def plot_q1c_diagnostics(B_list, oob_accs, disagreements, margins, selected_B=100, out_name="fig1_q1c_diagnostics.png"):
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    
    color1 = "#1f77b4" # Blue
    color2 = "#d62728" # Red
    color3 = "#2ca02c" # Green
    
    ax1.set_xlabel("Number of Estimators ($B$)")
    ax1.set_ylabel("Accuracy / Margin", color=color1)
    line1 = ax1.plot(B_list, oob_accs, marker="o", color=color1, label="OOB Accuracy")
    line3 = ax1.plot(B_list, margins, marker="^", color=color3, label="Mean Vote Margin")
    ax1.tick_params(axis="y", labelcolor=color1)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Pairwise Disagreement", color=color2)
    line2 = ax2.plot(B_list, disagreements, marker="s", color=color2, linestyle="--", label="Pairwise Disagreement")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False) # avoid cluttering
    
    # Highlight selected B
    ax1.axvline(x=selected_B, color="gray", linestyle=":", linewidth=1.5, label=f"Selected B={selected_B}")
    
    lines = line1 + line3 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.9)
    
    ax1.set_title("Bagging Diagnostics: OOB Accuracy, Disagreement & Margin vs. B")
    return save_fig(fig, out_name)

def plot_q2b_boosting_dynamics(weighted_errors, alphas, max_weights, top10_weights, out_name="fig3_q2b_boosting_dynamics.png"):
    rounds = np.arange(1, len(weighted_errors) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.2, 5.0))
    
    # 1. Weighted training error
    ax1.plot(rounds, weighted_errors, color="#d62728", marker="o", markersize=3)
    ax1.axhline(y=(40 - 1) / 40.0, color="gray", linestyle=":", label="Random Guess (0.975)")
    ax1.set_title(r"(a) Weighted Training Error ($\epsilon_t$)")
    ax1.set_xlabel("Boosting Round ($t$)")
    ax1.set_ylabel(r"$\epsilon_t$")
    ax1.legend(loc="upper left")
    
    # 2. Learner weight alpha_t
    ax2.plot(rounds, alphas, color="#1f77b4", marker="s", markersize=3)
    ax2.set_title(r"(b) Learner Weight ($\alpha_t$)")
    ax2.set_xlabel("Boosting Round ($t$)")
    ax2.set_ylabel(r"$\alpha_t$")
    
    # 3. Maximum sample weight
    ax3.plot(rounds, max_weights, color="#9467bd", marker="^", markersize=3)
    ax3.set_title(r"(c) Maximum Sample Weight ($\max_i w_i^{(t)}$)")
    ax3.set_xlabel("Boosting Round ($t$)")
    ax3.set_ylabel(r"$\max_i w_i^{(t)}$")
    
    # 4. Total weight carried by top-10 heaviest samples
    ax4.plot(rounds, top10_weights, color="#ff7f0e", marker="d", markersize=3)
    ax4.set_title("(d) Top-10 Heaviest Samples Weight")
    ax4.set_xlabel("Boosting Round ($t$)")
    ax4.set_ylabel(r"$\sum_{i \in Top10} w_i^{(t)}$")
    
    fig.suptitle("SAMME AdaBoost Training Dynamics over Boosting Rounds", fontsize=12)
    return save_fig(fig, out_name)

def plot_q3b_bias_variance(depth_labels, dt_bias, dt_var, bag_bias, bag_var, out_name="fig6_q3b_bias_variance.png"):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(depth_labels))
    width = 0.35
    
    # Stacked bars for Single Tree: Bias^2 + Variance
    p1 = ax.bar(x - width/2, dt_bias, width, label="Single Tree: $\\mathrm{Bias}^2$", color="#3182bd")
    p2 = ax.bar(x - width/2, dt_var, width, bottom=dt_bias, label="Single Tree: $\\mathrm{Variance}$", color="#9ecae1")
    
    # Stacked bars for 100-Tree Bagging: Bias^2 + Variance
    p3 = ax.bar(x + width/2, bag_bias, width, label="100-Tree Bagging: $\\mathrm{Bias}^2$", color="#31a354")
    p4 = ax.bar(x + width/2, bag_var, width, bottom=bag_bias, label="100-Tree Bagging: $\\mathrm{Variance}$", color="#a1d99b")
    
    ax.set_ylabel("Decomposed Error (Total MSE)")
    ax.set_title("Bias-Variance Decomposition vs. Tree Depth (Single Tree vs. Bagging)")
    ax.set_xticks(x)
    ax.set_xticklabels(depth_labels)
    ax.legend(loc="upper right", framealpha=0.9)
    
    return save_fig(fig, out_name)
