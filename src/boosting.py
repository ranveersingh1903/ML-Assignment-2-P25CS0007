"""
SAMME AdaBoost from scratch for Multiclass Classification.
Roll Number: P25CS0007
Student Name: Ranvir Kumar Singh
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SAMMEBoostingFromScratch:
    def __init__(self, max_depth=1, n_rounds=50, random_state=250007):
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.random_state = random_state
        self.estimators_ = []
        self.alphas_ = []
        self.weighted_errors_ = []
        self.sample_weight_history_ = []  # List of weights at each round
        self.classes_ = None
        self.n_classes_ = None
        self.stopped_early_ = False
        self.stop_reason_ = None

    def fit(self, X, y, sample_weight=None, log_history=True):
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        K = self.n_classes_
        
        # Initialize sample weights
        if sample_weight is None:
            w = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
        else:
            w = np.array(sample_weight, dtype=np.float64)
            w /= np.sum(w)
            
        rng = np.random.default_rng(self.random_state)
        
        self.estimators_ = []
        self.alphas_ = []
        self.weighted_errors_ = []
        self.sample_weight_history_ = []
        self.stopped_early_ = False
        self.stop_reason_ = None
        
        for t in range(self.n_rounds):
            if log_history:
                self.sample_weight_history_.append(w.copy())
                
            # Weak learner seed
            tree_seed = int(rng.integers(0, 1_000_000))
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=tree_seed
            )
            clf.fit(X, y, sample_weight=w)
            
            preds = clf.predict(X)
            incorrect = (preds != y)
            
            # Weighted training error
            err = np.sum(w * incorrect)
            
            # Explicit handling of weak learner not better than random multiclass prediction
            # Random guessing error is (K - 1) / K.
            random_error = (K - 1.0) / K
            
            if err >= random_error:
                self.stopped_early_ = True
                self.stop_reason_ = f"Round {t}: Weak learner error ({err:.4f}) >= random error ({random_error:.4f}). Alpha <= 0."
                # If this happens on round 0, we still need at least one learner, but for t > 0 we break.
                if t > 0:
                    break
                else:
                    # Keep with minimum alpha or break
                    alpha = 1e-6
                    self.estimators_.append(clf)
                    self.alphas_.append(alpha)
                    self.weighted_errors_.append(err)
                    break

            # Handle perfect weak learner (err == 0)
            if err <= 1e-12:
                err = 1e-12
                alpha = np.log((1.0 - err) / err) + np.log(K - 1.0)
                self.estimators_.append(clf)
                self.alphas_.append(alpha)
                self.weighted_errors_.append(0.0)
                # Perfect learner, can stop boosting
                self.stopped_early_ = True
                self.stop_reason_ = f"Round {t}: Perfect weak learner (err=0)."
                break
                
            # SAMME alpha formula:
            # alpha_t = log((1 - eps_t) / eps_t) + log(K - 1)
            alpha = float(np.log((1.0 - err) / err) + np.log(K - 1.0))
            
            if alpha <= 0:
                self.stopped_early_ = True
                self.stop_reason_ = f"Round {t}: Computed alpha {alpha:.4f} <= 0."
                break
                
            self.estimators_.append(clf)
            self.alphas_.append(alpha)
            self.weighted_errors_.append(float(err))
            
            # Weight update: w_i <- w_i * exp(alpha * 1[c_i != h_t(x_i)])
            w = w * np.exp(alpha * incorrect.astype(np.float64))
            w_sum = np.sum(w)
            if w_sum > 0:
                w /= w_sum
            else:
                w = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
                
        # Final weight snapshot
        if log_history:
            self.sample_weight_history_.append(w.copy())
            
        self.alphas_ = np.array(self.alphas_)
        self.weighted_errors_ = np.array(self.weighted_errors_)
        return self

    def predict_weighted_votes(self, X):
        """
        Computes S_k(x) = sum_t alpha_t * 1[h_t(x) == k]
        Returns matrix S of shape (n_samples, n_classes).
        """
        n_samples = X.shape[0]
        S = np.zeros((n_samples, self.n_classes_))
        
        for clf, alpha in zip(self.estimators_, self.alphas_):
            preds = clf.predict(X)
            for i, p in enumerate(preds):
                S[i, p] += alpha
                
        return S

    def predict_proba(self, X):
        """
        Normalized weighted vote q_k(x) = S_k(x) / sum_t alpha_t
        """
        S = self.predict_weighted_votes(X)
        alpha_sum = np.sum(self.alphas_)
        if alpha_sum > 0:
            return S / alpha_sum
        else:
            return np.full((X.shape[0], self.n_classes_), 1.0 / self.n_classes_)

    def predict(self, X):
        S = self.predict_weighted_votes(X)
        return np.argmax(S, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def get_prediction_confidence(self, X):
        """
        Winning probability q_{(1)}(x) = max_k q_k(x).
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)

def run_samme_toy_example():
    """
    Creates a 3-class example with at most 10 samples (N=6).
    Logs one complete SAMME round:
    - initial sample weights
    - weak learner predictions
    - weighted error
    - alpha_t
    - updated normalised weights
    """
    print("=" * 60)
    print("SAMME AdaBoost: Toy 3-Class Verification (N=6 samples, K=3)")
    print("=" * 60)
    
    # 6 samples, 2 features, 3 classes (0, 1, 2)
    X_toy = np.array([
        [1.0, 2.0],  # Class 0
        [1.5, 1.8],  # Class 0
        [5.0, 8.0],  # Class 1
        [5.5, 7.5],  # Class 1
        [8.0, 1.5],  # Class 2
        [8.5, 2.0],  # Class 2
    ])
    y_toy = np.array([0, 0, 1, 1, 2, 2])
    
    N = len(y_toy)
    K = 3
    initial_weights = np.full(N, 1.0 / N)
    
    print(f"Number of samples N = {N}, Number of classes K = {K}")
    print("Initial sample weights w_i^(1):")
    for i, w in enumerate(initial_weights):
        print(f"  Sample {i} (class {y_toy[i]}): w = {w:.6f}")
        
    # Fit weak learner with depth 1
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(X_toy, y_toy, sample_weight=initial_weights)
    preds = stump.predict(X_toy)
    
    print("\nWeak Learner Predictions h_1(x_i):")
    for i in range(N):
        status = "CORRECT" if preds[i] == y_toy[i] else "MISCLASSIFIED"
        print(f"  Sample {i}: true={y_toy[i]}, pred={preds[i]} -> [{status}]")
        
    incorrect = (preds != y_toy)
    weighted_err = np.sum(initial_weights * incorrect)
    random_err = (K - 1.0) / K
    
    alpha_1 = np.log((1.0 - weighted_err) / weighted_err) + np.log(K - 1.0)
    
    print(f"\nWeighted Error eps_1 = sum_i w_i * 1[y_i != h_1(x_i)] = {weighted_err:.6f}")
    print(f"Random Guessing Baseline (K-1)/K = (3-1)/3 = {random_err:.6f}")
    print(f"Condition: eps_1 ({weighted_err:.4f}) < random_err ({random_err:.4f}) -> Weak learner is better than random.")
    print(f"SAMME Alpha_1 = log((1 - eps_1) / eps_1) + log(K - 1) = {alpha_1:.6f}")
    
    # Weight update
    raw_weights = initial_weights * np.exp(alpha_1 * incorrect.astype(float))
    normalized_weights = raw_weights / np.sum(raw_weights)
    
    print("\nUpdated Normalised Weights w_i^(2):")
    for i in range(N):
        ratio = normalized_weights[i] / initial_weights[i]
        status = "Misclassified (Weight INCREASED)" if incorrect[i] else "Correct (Weight DECREASED)"
        print(f"  Sample {i}: w^(1)={initial_weights[i]:.4f} -> w^(2)={normalized_weights[i]:.4f} (factor {ratio:.2f}x) [{status}]")
        
    print(f"\nSum of normalized weights: {np.sum(normalized_weights):.6f}")
    print("=" * 60)
    
    return {
        "X_toy": X_toy,
        "y_toy": y_toy,
        "initial_weights": initial_weights,
        "predictions": preds,
        "weighted_error": weighted_err,
        "alpha": alpha_1,
        "updated_weights": normalized_weights
    }

if __name__ == "__main__":
    run_samme_toy_example()
