import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

def evaluate_strategy(strategy_name, groups, folds, pipeline , X, y):
    print(f"\n[VALIDATION] Starting evaluation strategy: {strategy_name}")
    metrics = {'MSE': [], 'Pearson': [], 'Spearman': []}
    
    if strategy_name not in folds:
        raise ValueError(f"[VALIDATION] Unknown strategy: {strategy_name}")
    fold_ids = folds[strategy_name].values if hasattr(folds[strategy_name], "values") else folds[strategy_name]
    unique_folds = np.unique(fold_ids)
    print(f" - Using predefined {len(unique_folds)} folds from synergy_score.csv")

    metrics = {'MSE': [], 'Pearson': [], 'Spearman': []}

    def _slice(X, idx):
        return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    for fold in unique_folds:
        train_idx = np.where(fold_ids != fold)[0]
        test_idx  = np.where(fold_ids == fold)[0]
        print(f"   Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

        X_tr = _slice(X, train_idx)
        X_te = _slice(X, test_idx)
        y_tr = y[train_idx]
        y_te = y[test_idx]

        # Training & Predicting
        print("   > Training model...", end="")
        pipeline.fit(X_tr, y_tr)
        print("Done! Predicting...", end="")
        y_pred = pipeline.predict(X_te)
        print("Done! Calculating metrics...")

        # calculate metrics
        mse      = mean_squared_error(y_te, y_pred)
        pearson  = pearsonr(y_te, y_pred)[0]
        spearman = spearmanr(y_te, y_pred)[0]
        metrics['MSE'].append(mse)
        metrics['Pearson'].append(pearson)
        metrics['Spearman'].append(spearman)

        print(f"   - MSE: {mse:.3f}, Pearson: {pearson:.3f}, Spearman: {spearman:.3f}")

    # Print aggregated metrics
    print(f"[VALIDATION] {strategy_name} evaluation completed! Aggregating metrics...")
    for name, vals in metrics.items():
        mean_val = np.mean(vals)
        std_val  = np.std(vals)
        print(f"   → {name}: {mean_val:.3f} ± {std_val:.3f}")

    return metrics

