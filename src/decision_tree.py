"""
Single Decision Tree Baseline exploration and evaluation.
Roll Number: P25CS0007
Student Name: Ranvir Kumar Singh
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.dataset import load_dataset, SEED

def get_leaf_metrics(dt: DecisionTreeClassifier):
    """
    Computes:
    - n_leaves: total number of leaf nodes
    - mean_samples_per_leaf: mean number of training samples in leaf nodes
    - singleton_leaf_fraction: fraction of leaves containing exactly one training sample
    """
    tree_ = dt.tree_
    # Leaf nodes are those where children_left == -1 (or children_right == -1)
    is_leaf = (tree_.children_left == -1)
    leaf_samples = tree_.n_node_samples[is_leaf]
    
    n_leaves = len(leaf_samples)
    mean_samples_per_leaf = float(np.mean(leaf_samples)) if n_leaves > 0 else 0.0
    singleton_leaves = np.sum(leaf_samples == 1)
    singleton_fraction = float(singleton_leaves / n_leaves) if n_leaves > 0 else 0.0
    
    return {
        "n_leaves": n_leaves,
        "mean_samples_per_leaf": mean_samples_per_leaf,
        "singleton_leaf_count": int(singleton_leaves),
        "singleton_fraction": singleton_fraction
    }

def run_decision_tree_grid(data_dir: str = "data"):
    data = load_dataset(data_dir)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test_clean"], data["y_test"]
    
    max_depth_options = [2, 4, 8, 16, None]
    min_samples_leaf_options = [1, 2, 4]
    
    results = []
    
    for md in max_depth_options:
        for msl in min_samples_leaf_options:
            # We use seed for reproducibility in tie-breaking splits
            clf = DecisionTreeClassifier(
                max_depth=md,
                min_samples_leaf=msl,
                random_state=SEED
            )
            clf.fit(X_train, y_train)
            
            train_acc = clf.score(X_train, y_train)
            val_acc = clf.score(X_val, y_val)
            test_acc = clf.score(X_test, y_test)
            
            leaf_metrics = get_leaf_metrics(clf)
            
            res = {
                "max_depth": str(md),
                "min_samples_leaf": msl,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "n_leaves": leaf_metrics["n_leaves"],
                "mean_samples_per_leaf": leaf_metrics["mean_samples_per_leaf"],
                "singleton_fraction": leaf_metrics["singleton_fraction"],
                "model": clf
            }
            results.append(res)
            
    df = pd.DataFrame(results)
    
    # Selection rule: Select ONE setting using validation accuracy only and freeze it.
    # Tie-breaking: If multiple have highest val_acc, prefer the one with highest min_samples_leaf / smaller leaves (less overfit)
    best_idx = df.sort_values(
        by=["val_acc", "min_samples_leaf"], 
        ascending=[False, False]
    ).index[0]
    
    best_setting = results[best_idx]
    
    print("\n--- Decision Tree Grid Results ---")
    print(df[["max_depth", "min_samples_leaf", "train_acc", "val_acc", "test_acc", "n_leaves", "mean_samples_per_leaf", "singleton_fraction"]].to_string())
    print("\n--- Selected Frozen Tree Setting (Tuned on Val Acc Only) ---")
    print(f"max_depth: {best_setting['max_depth']}, min_samples_leaf: {best_setting['min_samples_leaf']}")
    print(f"Train Acc: {best_setting['train_acc']:.4f}, Val Acc: {best_setting['val_acc']:.4f}, Test Acc: {best_setting['test_acc']:.4f}")
    print(f"Leaves: {best_setting['n_leaves']}, Mean Samples/Leaf: {best_setting['mean_samples_per_leaf']:.2f}, Singleton Frac: {best_setting['singleton_fraction']:.4f}")
    
    return df, best_setting

if __name__ == "__main__":
    run_decision_tree_grid()
