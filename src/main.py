from utils.data_loader import SynergyDataLoader,TranSynergyDataLoader
from utils.validation import evaluate_strategy
from utils.config import cfg
from models.random_forest import get_rf_model
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import argparse
import torch
from models.train_transynergy import run_training as run_transynergy

def parse_args():
    parser = argparse.ArgumentParser(description="Train TranSynergy or Random Forest on drug-combination data")
    parser.add_argument(
        "--model", 
        choices=["transynergy", "rf"],
        default="transynergy",
        help="Choose model: 'transynergy' (MLP) or 'rf' (Random Forest)")
    
    parser.add_argument(
        "--use_cv", 
        action="store_true",
        help="Whether to perform 5-fold cross-validation (TranSynergy only)")
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=128,
        help="Batch size for DataLoader (TranSynergy only)")
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of epochs for TranSynergy training")
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4,
        help="Learning rate for TranSynergy (TransSynergy only)")
    
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-5,
        help="Weight decay (L2) for TranSynergy optimizer")
    
    return parser.parse_args()


def main():
    #  Parsing command line arguments
    args = parse_args()
    # Set device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tp=cfg.TRANSYNERGY_PARAMS
    if args.batch_size is not None:
        tp["batch_size"] = args.batch_size
    if args.epochs is not None:
        tp["num_epochs"] = args.epochs
    if args.lr is not None:
        tp["lr"] = args.lr
    if args.weight_decay is not None:
        tp["weight_decay"] = args.weight_decay
    if args.use_cv:
        tp["use_cv"] = True

    
    tp["device"] = device

    print(f"[MAIN] Selected model: {args.model}")
    print(f"[MAIN] TransSynergy params: {tp}\n")

    if args.model == "transynergy":
        print("\n===== Starting TranSynergy Training(MLP) =====")
        # call train_transsynergy.py
        run_transynergy()
    else:
        print("\n===== Starting Random Forest Training =====")
        #  Parsing command line arguments
        parser = argparse.ArgumentParser(description="Train/Evaluate or Explain the RF model")
        parser.add_argument(
            "--mode", choices=["all","evaluate","explain"], default="all",
            help="Model: all(Train/Evaluate + Explainability Analysis)、evaluate(Only Train/ Evaluate)、explain(Only Explainability Analysis)"
        )

        args = parser.parse_args()
        if args.mode not in ["all", "evaluate", "explain"]:
            raise ValueError(f"Invalid mode: {args.mode}. Choose from 'all', 'evaluate', or 'explain'.")
        print(f"[MAIN] Running in {args.mode} mode.")

        print("\n===== Initializing System =====")
        print("> Creating data loader...")
        data_loader = SynergyDataLoader()
        X, y= data_loader.X, data_loader.y
        groups, folds = data_loader.groups, data_loader.folds
        feature_names = data_loader.feature_names
        print(f"[DATA] Loaded {len(y)} samples with {X.shape[1]} features.\n")

        print("\n> Building model pipeline...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components = cfg.PCA_COMPONENTS)),
            ("rf",     get_rf_model())
        ])
        print("[MODEL] Pipeline steps:", pipeline.named_steps, "\n")

        if args.mode in ["all", "evaluate"]:
            print("\n===== Starting Evaluation =====")
            strategies = ['drug_combo', 'cell_line', 'drug']
            summary = []
            for strategy in strategies:
                metrics = evaluate_strategy(strategy, groups, folds, pipeline, X, y)

                mean_mse     = np.mean(metrics["MSE"])
                std_mse      = np.std(metrics["MSE"])
                mean_pearson = np.mean(metrics["Pearson"])
                std_pearson  = np.std(metrics["Pearson"])
                mean_spear   = np.mean(metrics["Spearman"])
                std_spear    = np.std(metrics["Spearman"])

                print(f"\n[{strategy.upper()} RESULTS]")
                print(f" - Mean MSE:     {mean_mse:.1f} ± {std_mse:.1f}")
                print(f" - Mean Pearson: {mean_pearson:.3f} ± {std_pearson:.3f}")
                print(f" - Mean Spearman:{mean_spear:.3f} ± {std_spear:.3f}")

                summary.append({
                    'strategy':      strategy,
                    'mean_MSE':      mean_mse,
                    'std_MSE':       std_mse,
                    'mean_Pearson':  mean_pearson,
                    'std_Pearson':   std_pearson,
                    'mean_Spearman': mean_spear,
                    'std_Spearman':  std_spear
                })
            
            df_sum = pd.DataFrame(summary)
            out_path = "metrics_summary.csv"
            df_sum.to_csv(out_path, index=False)
            print(f"\n[MAIN] Saved summary metrics to {out_path}\n")
        
        if args.mode in ["all", "explain"]:
            print("\n===== Starting Explainability Analysis =====\n")
            # 1) Full Retraining Pipeline
            print("[EXPLAIN] Original feature Explainability analysis on 5000 dataset …")    
            sub_n =5000
            idx_sub = np.random.choice(len(X), size=sub_n, replace=False)
            X_sub   = X.iloc[idx_sub] if hasattr(X, "iloc") else X[idx_sub]
            y_sub   = y[idx_sub]
            pipe_orig = Pipeline([("scaler", StandardScaler()), ("rf", get_rf_model())])
            #print(f"[EXPLAIN] Subset shape: {X_sub.shape}, {len(y_sub)} samples")
            print(f"Model retraining on {sub_n} samples with {X_sub.shape[1]} features ...")
            pipe_orig.fit(X_sub, y_sub)
            
            # 2) Permutation Importance
            
            print("[EXPLAIN] Perm-Imp on 5k samples & full features")

            r = permutation_importance(
                pipe_orig, 
                X_sub, 
                y_sub, 
                n_repeats=3, 
                random_state=0, 
                n_jobs=-1)
            
            # im, istd = r.importances_mean, r.importances_std
            # order   = im.argsort()[::-1][:20]  # Top 20
            # for i in order:
            #     print(f"  {X_sub.columns[i]:<20s} {im[i]:.2f} ± {istd[i]:.2f}")

            imp_mean, imp_std = r.importances_mean, r.importances_std
            order = imp_mean.argsort()[::-1][:10]
            print("\n[EXPLAIN] Top 10 features by Perm-Imp:")
            for i in order:
                print(f"  {feature_names[i]:<20s} {imp_mean[i]:.4f} ± {imp_std[i]:.4f}")


            # 3) SHAP Values

            print("[EXPLAIN] SHAP on same 5k x 500 subset")
            expl = shap.TreeExplainer(pipe_orig.named_steps["rf"])
            Xs = pipe_orig.named_steps["scaler"].transform(X_sub)
            #sv = expl.shap_values(Xs)
            inx500 = np.random.choice(Xs.shape[0], size=500, replace=False)
            sha_vals = expl.shap_values(Xs[inx500])
            print(f"[EXPLAIN] SHAP values shape: {sha_vals.shape} for {len(inx500)} samples")

            #sv_norm = sv/ np.max(np.abs(sv))


            shap.summary_plot(
                sha_vals, 
                Xs[inx500],
                feature_names=feature_names, 
                max_display=20, 
                show=False)
            plt.tight_layout()
            plt.savefig("shap_500_featurename2.png", bbox_inches="tight")
            print("Saved fast SHAP → shap_500_featurename2.png")


if __name__ == "__main__":
    main()
