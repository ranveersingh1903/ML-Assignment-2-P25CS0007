"""
Bagging Classifier from scratch for Olivetti Faces.
Roll Number: P25CS0007
Student Name: Ranvir Kumar Singh
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class BaggingFromScratch:
    def __init__(self, base_params=None, n_estimators=10, random_state=250007):
        self.base_params = base_params or {"max_depth": None, "min_samples_leaf": 1}
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.bootstrap_indices_ = []
        self.oob_indices_ = []
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        rng = np.random.default_rng(self.random_state)
        
        self.estimators_ = []
        self.bootstrap_indices_ = []
        self.oob_indices_ = []
        
        all_indices = np.arange(n_samples)
        
        for b in range(self.n_estimators):
            # Bootstrap sample of size N with replacement
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            oob_idx = all_indices[oob_mask]
            
            tree_seed = int(rng.integers(0, 1_000_000))
            clf = DecisionTreeClassifier(
                **self.base_params,
                random_state=tree_seed
            )
            clf.fit(X[boot_idx], y[boot_idx])
            
            self.estimators_.append(clf)
            self.bootstrap_indices_.append(boot_idx)
            self.oob_indices_.append(oob_idx)
            
        return self

    def predict_tree_votes(self, X):
        """
        Returns an array of shape (n_estimators, n_samples) with class predictions.
        """
        preds = np.array([tree.predict(X) for tree in self.estimators_])
        return preds

    def predict_proba(self, X):
        """
        Returns vote fraction matrix of shape (n_samples, n_classes).
        """
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes_))
        preds = self.predict_tree_votes(X) # shape (B, N)
        for b in range(self.n_estimators):
            for i in range(n_samples):
                c = preds[b, i]
                votes[i, c] += 1
        return votes / self.n_estimators

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def get_average_distinct_samples(self):
        """
        Average number of distinct training samples across bootstrap samples.
        """
        distinct_counts = [len(np.unique(idx)) for idx in self.bootstrap_indices_]
        return float(np.mean(distinct_counts)), distinct_counts

    def compute_oob_accuracy(self, X_train, y_train):
        """
        Computes out-of-bag accuracy on training set.
        """
        n_samples = X_train.shape[0]
        oob_votes = np.zeros((n_samples, self.n_classes_))
        oob_counts = np.zeros(n_samples, dtype=int)
        
        for b in range(self.n_estimators):
            oob_idx = self.oob_indices_[b]
            if len(oob_idx) == 0:
                continue
            preds = self.estimators_[b].predict(X_train[oob_idx])
            for idx, pred in zip(oob_idx, preds):
                oob_votes[idx, pred] += 1
                oob_counts[idx] += 1
                
        valid_oob = oob_counts > 0
        if not np.any(valid_oob):
            return 0.0, 0
            
        oob_preds = np.argmax(oob_votes[valid_oob], axis=1)
        oob_acc = float(np.mean(oob_preds == y_train[valid_oob]))
        return oob_acc, int(np.sum(valid_oob))

    def compute_pairwise_disagreement(self, X):
        """
        Average pairwise disagreement between trees on sample set X:
        (1 / (B * (B - 1) / 2)) * sum_{b1 < b2} mean(pred_b1 != pred_b2)
        Vectorized exact calculation:
        agreeing_pairs_i = sum_k choose(c_{i, k}, 2) = 0.5 * (sum_k c_{i, k}^2 - B)
        disagree_i = 1 - (sum_k c_{i, k}^2 - B) / (B * (B - 1)) = (B^2 - sum_k c_{i, k}^2) / (B * (B - 1))
        """
        B = self.n_estimators
        if B < 2:
            return 0.0
            
        proba = self.predict_proba(X) # (N, K)
        counts = proba * B            # (N, K)
        sum_sq = np.sum(counts ** 2, axis=1) # (N,)
        disagree_i = (B ** 2 - sum_sq) / (B * (B - 1.0))
        return float(np.mean(disagree_i))

    def compute_mean_vote_margin(self, X):
        """
        Mean vote margin on X: winning vote fraction minus runner up vote fraction.
        """
        proba = self.predict_proba(X) # shape (N, K)
        sorted_proba = np.sort(proba, axis=1) # ascending
        margins = sorted_proba[:, -1] - sorted_proba[:, -2]
        return float(np.mean(margins)), margins

    def get_prediction_confidence(self, X):
        """
        Winning vote fraction for each sample in X.
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)
