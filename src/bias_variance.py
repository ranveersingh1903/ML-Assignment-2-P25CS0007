"""
Empirical Bias-Variance estimation module.
Roll Number: P25CS0007
Student Name: Ranvir Kumar Singh
"""

import numpy as np

def generate_resampled_train_sets(X_train, y_train, n_identities=40, n_per_id=6, R=30, seed=250007):
    """
    Creates R new training sets by sampling with replacement, within each identity,
    6 faces from that identity's original 6 training faces.
    """
    rng = np.random.default_rng(seed)
    resampled_sets = []
    
    # Map identity to its original training sample indices
    id_to_indices = {}
    for c in range(n_identities):
        id_to_indices[c] = np.where(y_train == c)[0]
        assert len(id_to_indices[c]) == n_per_id, f"Identity {c} has {len(id_to_indices[c])} samples, expected {n_per_id}"
        
    for r in range(R):
        resampled_idx = []
        for c in range(n_identities):
            orig_idx = id_to_indices[c]
            # Sample 6 with replacement from original 6
            chosen = rng.choice(orig_idx, size=n_per_id, replace=True)
            resampled_idx.extend(chosen)
            
        resampled_idx = np.array(resampled_idx)
        X_r = X_train[resampled_idx]
        y_r = y_train[resampled_idx]
        resampled_sets.append((X_r, y_r, resampled_idx))
        
    return resampled_sets

def compute_bias_variance(p_r_all, y_test, n_classes=40):
    """
    Computes Bias^2 and Variance according to:
    Bias^2 = (1 / N) * sum_i || p_bar(x_i) - e_i ||_2^2
    Variance = (1 / (N * R)) * sum_{i, r} || p_r(x_i) - p_bar(x_i) ||_2^2
    
    Parameters:
    - p_r_all: array of shape (R, N, K)
    - y_test: array of shape (N,)
    - n_classes: K (40 for Olivetti)
    """
    R, N, K = p_r_all.shape
    assert len(y_test) == N
    assert K == n_classes
    
    # e_i is the one-hot target
    E = np.zeros((N, K), dtype=np.float64)
    for i, target in enumerate(y_test):
        E[i, target] = 1.0
        
    # p_bar(x_i) = mean over repetitions
    p_bar = np.mean(p_r_all, axis=0) # shape (N, K)
    
    # Bias^2 = (1 / N) * sum_i || p_bar(x_i) - e_i ||_2^2
    diff_bias = p_bar - E # (N, K)
    bias_squared = float(np.mean(np.sum(diff_bias ** 2, axis=1)))
    
    # Variance = (1 / (N * R)) * sum_{i, r} || p_r(x_i) - p_bar(x_i) ||_2^2
    # p_r_all - p_bar[None, :, :] has shape (R, N, K)
    diff_var = p_r_all - p_bar[np.newaxis, :, :]
    variance = float(np.mean(np.sum(diff_var ** 2, axis=2)))
    
    # Verification of total MSE
    diff_total = p_r_all - E[np.newaxis, :, :]
    total_mse = float(np.mean(np.sum(diff_total ** 2, axis=2)))
    decomposition_error = abs(total_mse - (bias_squared + variance))
    
    return {
        "bias_squared": bias_squared,
        "variance": variance,
        "total_mse": total_mse,
        "decomposition_error": decomposition_error,
        "p_bar": p_bar
    }
