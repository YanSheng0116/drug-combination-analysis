# scripts/transynergy_postprocess.py

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import make_pipeline
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from utils.config import cfg
from utils.data_loader import TranSynergyDataLoader
from models.transynergy_net import TranSynergyNet


def plot_learning_curves(cv_mode: bool, num_epochs: int, cv_folds: int):
    """
    If cv_mode is False, look for a single learning curve image:
        'transynergy_learning_curve.png'
    If cv_mode is True, look for per-fold curves:
        'learning_curve_fold1.png', ..., 'learning_curve_fold{cv_folds}.png'
    Simply list which files are present or missing.
    """
    if cv_mode:
        print("\n[POST] Learning curves in CV mode:")
        for i in range(1, cv_folds + 1):
            fname = f"results/learning_curve_fold{i}.png"
            if os.path.exists(fname):
                print(f"  ✔ Fold {i}: {fname}")
            else:
                print(f"  ✘ Fold {i} learning curve file {fname} not found!")
    else:
        fname = "reults/transynergy_learning_curve.png"
        print("\n[POST] Learning curve in single-run mode:")
        if os.path.exists(fname):
            print(f"  ✔ {fname}")
        else:
            print(f"  ✘ Learning curve file {fname} not found!")


def summarize_cv_results():
    """
    Read 'transynergy_cv_results.csv' (per-fold test metrics),
    compute mean and standard deviation for each metric,
    save a summary CSV: 'transynergy_cv_summary.csv',
    and plot a bar chart (mean ± std) named:
        'transynergy_cv_summary_bar.png'
    """
    src_csv = "results/transynergy_cv_results.csv"
    if not os.path.exists(src_csv):
        print(f"[POST][ERROR] CV results file '{src_csv}' not found. Run training with --use_cv first.")
        return

    df = pd.read_csv(src_csv)
    metrics = ["test_mse", "test_mae", "test_r2", "test_pearson", "test_spearman"]
    means = df[metrics].mean()
    stds = df[metrics].std()

    df_summary = pd.DataFrame({
        "metric": metrics,
        "mean": means.values,
        "std": stds.values
    })
    df_summary.to_csv("results/transynergy_cv_summary.csv", index=False)
    print(f"[POST] Saved CV summary metrics → 'transynergy_cv_summary.csv'")

    # Plot bar chart with error bars
    plt.figure(figsize=(8, 4))
    sns.set_style("whitegrid")
    x_pos = np.arange(len(metrics))
    plt.bar(x_pos, means.values, yerr=stds.values, capsize=5, alpha=0.7, color="skyblue")
    plt.xticks(x_pos, ["MSE", "MAE", "R2", "Pearson", "Spearman"])
    plt.ylabel("Value")
    plt.title("TransSynergy 5-Fold CV Results (mean ± std)")
    plt.tight_layout()
    out_png = "results/transynergy_cv_summary_bar.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[POST] Saved CV summary bar chart → '{out_png}'")


def shap_feature_importance(topk: int = 20):
    """
    Perform SHAP-based feature importance on the full dataset:
      1) Load pre_normalized X_all, y_all from TranSynergyDataLoader
      2) Build model architecture and load final weights 'weights/model_weights.pth'
      (Loads pre-trained TranSynergyNet neural network. Uses saved weights from model_weights.pth)
      3) Use DeepExplainer to compute SHAP values
      4) For each feature block (RWR-drugA, RWR-drugB, Expression, Dependency),
         select topk genes by mean(|SHAP|), and plot horizontal bar charts:
           - 'shap_top{topk}_RWR_drugA.png'
           - 'shap_top{topk}_RWR_drugB.png'
           - 'shap_top{topk}_Expression.png'
           - 'shap_top{topk}_Dependency.png'
    """
    # 1) Load normalized feature matrix and labels
    loader_rep = TranSynergyDataLoader()
    X_all = loader_rep.X  # shape = (n_samples, 9608)
    y_all = loader_rep.y  # shape = (n_samples,)

    # 2) Load gene list to reconstruct feature names
    syn_loader = TranSynergyDataLoader()
    gene_list = syn_loader.gene_list  # length = 2401

    # Define column index blocks for each feature type
    n_genes = len(gene_list)
    idx_rwrA = slice(0, n_genes)
    idx_rwrB = slice(n_genes, 2 * n_genes)
    idx_dep = slice(2 * n_genes, 3 * n_genes)
    idx_expr = slice(3 * n_genes, 4 * n_genes)
    # The remaining 4 columns are metadata, ignored for SHAP

    # 3) Build model and load saved weights
    model = TranSynergyNet()
    model_path = "weights/model_weights.pth"
    if not os.path.exists(model_path):
        print(f"[POST][ERROR] Model weights '{model_path}' not found. Complete single-run training first.")
        return
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 4) Sample background data for SHAP
    np.random.seed(cfg.TRANSYNERGY_PARAMS["seed"])
    n_background = min(5000, X_all.shape[0])
    idx_bg = np.random.choice(X_all.shape[0], n_background, replace=False)
    X_bg = torch.tensor(X_all[idx_bg], dtype=torch.float32)
    n_explain = min(1000, X_all.shape[0])
    X_explain_np= X_all[:n_explain]


    print(f"[POST] Computing SHAP values with DeepExplainer, background size = {n_background}, explain size = {n_explain}")
    # define a prediction function that takes numpy and returns numpy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def f_np(x: np.ndarray) -> np.ndarray:
        xt = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            y = model(xt).cpu().numpy()
        return y.reshape(-1)

    X_bg_tensor = torch.tensor(X_bg, dtype=torch.float32).to(device)

    explainer = shap.DeepExplainer(model,X_bg_tensor)
    batch_size =10
    shap_values_list= []
    for i in range(0, n_explain, batch_size):
        batch_data = X_explain_np[i:i+batch_size]
        X_explain_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
        
        # caculate shap value in batch
        batch_shap = explainer.shap_values(X_explain_tensor)

        shap_values_list.append(batch_shap[0])

    shap_values = np.vstack(shap_values_list)


    # 6) Compute mean(|SHAP|) across explain samples
    shap_mean = np.mean(np.abs(shap_values), axis=0)
    print(f"[POST][DEBUG] SHAP mean shape: {shap_mean.shape}")
    print(f"[POST][DEBUG] SHAP values shape: {shap_values.shape}")


    # 7) Helper to get topk indices within a block
    def get_topk_indices(block_slice, block_name):
        arr = shap_mean[block_slice]
        local_sorted = np.argsort(-arr)  # descending order
        top_local = local_sorted[:topk]
        top_global = [block_slice.start + i for i in top_local]
        return top_global

    topA = get_topk_indices(idx_rwrA, "RWR_drugA")
    topB = get_topk_indices(idx_rwrB, "RWR_drugB")
    topE = get_topk_indices(idx_expr, "Expression")
    topD = get_topk_indices(idx_dep, "Dependency")

    # 8) Plot topk genes for each block
    def plot_block_topk(top_indices, block_name):
        if block_name == "RWR_drugA":
            offset = idx_rwrA.start
        elif block_name == "RWR_drugB":
            offset = idx_rwrB.start
        elif block_name == "Expression":
            offset = idx_expr.start
        elif block_name == "Dependency":
            offset = idx_dep.start
        else:
            raise ValueError("Unknown block name")

        gene_names = [gene_list[i - offset] for i in top_indices]
        shap_vals = [shap_mean[i] for i in top_indices]

        df_plot = pd.DataFrame({
            "gene": gene_names,
            "shap_value": shap_vals
        }).sort_values("shap_value", ascending=True)

        plt.figure(figsize=(6, max(4, topk * 0.25)))
        sns.barplot(x="shap_value", y="gene", data=df_plot, palette="viridis")
        plt.title(f"Top {topk} SHAP Scores | {block_name}")
        plt.xlabel("Mean |SHAP value|")
        plt.ylabel("Gene")
        plt.tight_layout()
        out_png = f"results/shap_top{topk}_{block_name}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[POST] Saved {block_name} SHAP top-{topk} plot → '{out_png}'")

    plot_block_topk(topA, "RWR_drugA")
    plot_block_topk(topB, "RWR_drugB")
    plot_block_topk(topE, "Expression")
    plot_block_topk(topD, "Dependency")

    feature_names = syn_loader.feature_names
    n_show = min(shap_values.shape[0],200)

    idx_plot = np.random.choice(shap_values.shape[0], size=n_show, replace=False)
    sv_plot = shap_values[idx_plot]
    X_plot  = X_explain_np[idx_plot]
    gene_cols = [c for c in feature_names if not c.endswith("fold")]
    idx = [feature_names.index(c) for c in gene_cols]
    sv_gene = sv_plot[:, idx]
    X_gene  = X_plot[:, idx]
    shap.summary_plot(sv_gene, X_gene, feature_names=gene_cols, max_display=20, show=False)
    plt.title("SHAP Summary Beeswarm Plot (Top 20 Features)")
    plt.tight_layout()
    plt.savefig("results/shap_summary_beeswarm.png", bbox_inches="tight")
    print("Saved → shap_summary_beeswarm.png")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-processing for TranSynergy: plot learning curves, summarize CV, and SHAP analysis"
    )
    parser.add_argument(
        "--use_cv",
        action="store_true",
        help="Whether to summarize CV results (expect 'transynergy_cv_results.csv' to exist)."
    )
    parser.add_argument(
        "--shap_only",
        action="store_true",
        help="If set, skip learning-curve/CV summary and only run SHAP analysis."
    )
    args = parser.parse_args()

    # 1) Plot learning curves if not shap_only
    if not args.shap_only:
        plot_learning_curves(
            cv_mode=args.use_cv,
            num_epochs=cfg.TRANSYNERGY_PARAMS["num_epochs"],
            cv_folds=cfg.TRANSYNERGY_PARAMS["cv_folds"]
        )

    # 2) Summarize CV results if requested
    if args.use_cv and not args.shap_only:
        summarize_cv_results()

    # 3) Run SHAP analysis (always run, or if --shap_only)
    print("\n[POST] Starting SHAP analysis (this may take a while)…")
    shap_feature_importance(topk=20)

    print("\n[POST] Post-processing complete. All CSVs and PNGs have been generated.\n")
