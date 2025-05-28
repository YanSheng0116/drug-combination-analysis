from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

def calculate_metrics(y_true, y_pred):
    """compute MSE、Pearson、Spearman"""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'Pearson': pearsonr(y_true, y_pred)[0],
        'Spearman': spearmanr(y_true, y_pred)[0]
    }