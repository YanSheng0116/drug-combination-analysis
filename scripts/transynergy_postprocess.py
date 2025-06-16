# scripts/transynergy_postprocess.py

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline
import gseapy as gp

import shap
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

def load_data():
    """
    Load normalized feature matrix X_all, labels y_all, and feature_names.
    """
    loader = TranSynergyDataLoader()
    X_all = loader.X     # (n_samples, 9608)
    y_all = loader.y     # (n_samples,)
    feature_names = loader.feature_names
    return X_all, y_all, feature_names

def load_model(weights_path: str, device='cpu'):
    """
    Instantiate TranSynergyNet, load weights, set to eval.
    """
    model = TranSynergyNet()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[POST][ERROR] Model weights '{weights_path}' not found.")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model

def shap_feature_importance(topk: int = 20):
    """
    Perform SHAP-based feature importance on the full dataset:
      1) Load pre_normalized X_all, y_all from TranSynergyDataLoader
      2) Build model architecture and load final weights
      3) Use DeepExplainer to compute SHAP values in batches
      4) For each feature block, select topk genes by mean(|SHAP|), plot bar charts
      5) Finally, plot a beeswarm summary of the top 20 gene features
    """
    # 1) Load data
    loader_rep = TranSynergyDataLoader()
    X_all = loader_rep.X
    y_all = loader_rep.y

    # 2) Load gene list & define blocks
    gene_list = loader_rep.gene_list
    n_genes = len(gene_list)
    idx_rwrA = slice(0, n_genes)
    idx_rwrB = slice(n_genes, 2*n_genes)
    idx_dep  = slice(2*n_genes, 3*n_genes)
    idx_expr = slice(3*n_genes, 4*n_genes)
    # metadata at the end

    # 3) Build & load model
    model_path = "weights/model_weights.pth"
    model = load_model(model_path, device="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 4) Sample background & explain sets
    np.random.seed(cfg.TRANSYNERGY_PARAMS["seed"])
    n_background = min(500, X_all.shape[0])
    idx_bg = np.random.choice(X_all.shape[0], n_background, replace=False)
    X_bg_np = X_all[idx_bg]
    bg_data = shap.kmeans(X_bg_np,100)
    X_bg_protos= bg_data.data if hasattr(bg_data, 'data') else bg_data
    X_bg_tensor = torch.tensor(X_bg_protos, dtype=torch.float32).to(device)
    n_explain = min(200, X_all.shape[0])
    X_explain_np = X_all[:n_explain]
    X_explain_tensor = torch.tensor(X_explain_np, dtype=torch.float32).to(device)

    print(f"[POST] Computing SHAP values with DeepExplainer, background size = {n_background}, explain size = {n_explain}")
    # define numpy wrapper
    def f_np(x: np.ndarray) -> np.ndarray:
        xt = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            y = model(xt).cpu().numpy()
        return y.reshape(-1)

    # create DeepExplainer
    explainer = shap.DeepExplainer(model, X_bg_tensor)

    # 5) Compute SHAP values
    runs = []
    for _ in range(10):

        out = explainer.shap_values(X_explain_tensor)
        run = out[0] if isinstance(out, list) else out  # handle single output case
        runs.append(np.array(run))  # (n_explain, 9608)
    # stack runs to shape (10, n_explain, 9608) 
    shap_values = np.mean(np.stack(runs, axis=0), axis=0)  # average over repeats

    # 7) Compute mean(|SHAP|) over explain samples
    shap_mean = np.mean(np.abs(shap_values), axis=0)

    print(f"[POST][DEBUG] SHAP mean shape: {shap_mean.shape}")
    print(f"[POST][DEBUG] SHAP values shape: {shap_values.shape}")

    # 7) Get topk for each block
    def get_topk_indices(block_slice):
        arr = shap_mean[block_slice]
        top_local = np.argsort(-arr)[:topk]
        return [block_slice.start + i for i in top_local]

    topA = get_topk_indices(idx_rwrA)
    topB = get_topk_indices(idx_rwrB)
    topE = get_topk_indices(idx_expr)
    topD = get_topk_indices(idx_dep)

    # 8) Plot block bar charts
    def plot_block_topk(top_indices, block_name, block_slice):
        offset = block_slice.start
        names = [gene_list[i-offset] for i in top_indices]
        vals  = [shap_mean[i] for i in top_indices]
        df = pd.DataFrame({"gene": names, "shap": vals}).sort_values("shap", ascending=True)

        plt.figure(figsize=(6, max(4, topk*0.3)))
        sns.barplot(x="shap", y="gene", data=df, palette="viridis")
        plt.title(f"Top {topk} SHAP Scores | {block_name}")
        plt.xlabel("Mean |SHAP value|")
        plt.tight_layout()
        out = f"results/shap_top{topk}_{block_name}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[POST] Saved {block_name} SHAP top-{topk} plot → '{out}'")

    plot_block_topk(topA, "RWR_drugA", idx_rwrA)
    plot_block_topk(topB, "RWR_drugB", idx_rwrB)
    plot_block_topk(topE, "Expression", idx_expr)
    plot_block_topk(topD, "Dependency", idx_dep)

    # 9) Beeswarm summary (genes only)
    feature_names = loader_rep.feature_names
    gene_cols = [c for c in feature_names if not c.endswith("fold")]
    idx_genes = [feature_names.index(c) for c in gene_cols]
    n_show = min(shap_values.shape[0], 200)
    idx_plot = np.random.choice(shap_values.shape[0], size=n_show, replace=False)
    sv_plot = shap_values[idx_plot][:, idx_genes]
    X_plot  = X_explain_np[idx_plot][:, idx_genes]

    shap.summary_plot(
        sv_plot, X_plot,
        feature_names=gene_cols,
        max_display=20,
        show=False
    )
    plt.title("SHAP Summary Beeswarm Plot (Top 20 Features)")
    plt.tight_layout()
    out = "results/shap_summary_beeswarm.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"[POST] Saved → {out}")

    # 10) Prepare for GSEA
    shap_gene = shap_mean[:4 * len(gene_list)]  # only take 4x2401  RWR/DEP/EXPR 
    # If you want to perform GSEA separately for RWR_drugA and RWR_drugB, slice them individually:
    blocks = {
        "RWR_drugA": shap_gene[idx_rwrA],
        "RWR_drugB": shap_gene[idx_rwrB],
        "Dependency": shap_gene[idx_dep],
        "Expression": shap_gene[idx_expr]
    }

    # 11) For each block, run prerank GSEA
    for block_name, values in blocks.items():
        # Construct rank table: gene_name -> shap value
        rnk = pd.DataFrame({
            "gene": gene_list,
            "shap": values
        }).sort_values("shap", ascending=False)

        # Call gseapy.prerank
        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets="data/gene_sets/c6.all.v7.5.1.symbols.gmt",     #  MSigDB Oncogenic Signatures C6 .gmt file path
            processes=4,
            permutation_num=100,        # Number of permutations, can be adjusted based on data volume
            outdir=f"results/gsea_{block_name}", 
            format='png',
            seed=cfg.TRANSYNERGY_PARAMS["seed"],
            verbose=True
        )

        # Save top 10 significant pathways to CSV English:
        df_term = pre_res.res2d.reset_index().rename(columns={"index":"Term"})
        out_csv = f"results/gsea_{block_name}_top10.csv"
        df_term.head(10).to_csv(out_csv, index=False)
        print(f"[POST] Saved GSEA results for {block_name} → {out_csv}")

    print("[POST] SA‐GSEA complete.")

    for block_name in ["RWR_drugA", "RWR_drugB", "Dependency", "Expression"]:
        csv_path = f"results/gsea_{block_name}_top10.csv"
        df = pd.read_csv(csv_path)
        # 1) 统一小写列名
        df.columns = df.columns.str.lower()
        # 2) 确保有 term 列
        if 'term' not in df.columns:
            df = df.rename(columns={df.columns[0]: 'term'})

        # 3) Barplot: NES
        if 'nes' not in df.columns:
            raise ValueError(f"Expected 'nes' column in {csv_path}, got {df.columns.tolist()}")
        plt.figure(figsize=(6,4))
        sns.barplot(x="nes", y="term", data=df, palette="magma", orient="h")
        plt.xlabel("Normalized Enrichment Score (NES)")
        plt.ylabel("")
        plt.title(f"GSEA Top10 NES | {block_name}")
        plt.tight_layout()
        out_bar = f"results/gsea_{block_name}_NES_bar.png"
        plt.savefig(out_bar, dpi=150)
        plt.close()
        print(f"[POST] Saved GSEA NES barplot → {out_bar}")

        # 4) Bubble Plot: only if we have a size column
        size_cols = [c for c in df.columns if 'size' in c]
        if size_cols and 'fdr' in df.columns:
            size_col = size_cols[0]
            hue_vals = -np.log10(df['fdr'])
            plt.figure(figsize=(6,4))
            sns.scatterplot(
                x="nes", y="term",
                size=size_col, sizes=(50,300),
                hue=hue_vals, palette="viridis",
                legend='brief', data=df
            )
            plt.xlabel("Normalized Enrichment Score (NES)")
            plt.ylabel("")
            plt.legend(loc="lower right", title="-log10(FDR)")
            plt.title(f"GSEA Bubble Plot | {block_name}")
            plt.tight_layout()
            out_bubble = f"results/gsea_{block_name}_bubble.png"
            plt.savefig(out_bubble, dpi=150)
            plt.close()
            print(f"[POST] Saved GSEA bubble plot → {out_bubble}")
        else:
            print(f"[WARN] No size/fdr columns in {csv_path}, skipping bubble plot.")

def lime_feature_importance(model, X_all, feature_names, explain_size=10, topk=20):
    """
    Local LIME explanations on first few samples.
    """
    print("[XAI][LIME] Starting LIME analysis…")
    def f_np(x): return model(torch.tensor(x, dtype=torch.float32)).detach().numpy().reshape(-1)
    explainer = LimeTabularExplainer(
        X_all, feature_names=feature_names, verbose=False, mode='regression'
    )
    for i in range(min(explain_size, len(X_all))):
        exp = explainer.explain_instance(
            X_all[i], f_np, num_features=topk, num_samples=500
        )
        fig = exp.as_pyplot_figure()
        fig.suptitle(f"LIME sample {i}", y=1.02)
        fig.tight_layout()
        fn = f"results/lime_sample_{i}.png"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        print(f"[XAI][LIME] Saved → {fn}")

def anchor_feature_importance(model, X_all, feature_names, explain_size=10):
    """
    Anchors explanations on first few samples.
    """
    
    print("[XAI][ANCHOR] Starting Anchors analysis…")
    from alibi.explainers import AnchorTabular
    def f_np(x): return model(torch.tensor(x, dtype=torch.float32)).detach().numpy().reshape(-1)
    explainer = AnchorTabular(
        predictor=f_np,
        feature_names=feature_names,
        categorical_names={}
    )
    for i in range(min(explain_size, len(X_all))):
        exp = explainer.explain(X_all[i], threshold=0.95)
        html = exp.as_html()
        fn = f"results/anchor_sample_{i}.html"
        with open(fn, 'w') as f:
            f.write(html)
        print(f"[XAI][ANCHOR] Saved → {fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-processing for TranSynergy: learning curves, CV summary & XAI"
    )
    parser.add_argument(
        "--use_cv", action="store_true",
        help="Whether to summarize CV results (expect CSV to exist)."
    )
    parser.add_argument(
        "--shap_only", action="store_true",
        help="If set, skip learning-curve/CV and only run XAI."
    )
    parser.add_argument(
        "--xai",
        choices=['shap','lime','anchor','all'],
        default='shap',
        help="Which explainability method(s) to run."
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to TranSynergyNet weights (.pth)."
    )
    args = parser.parse_args()

    # 1) Learning curves & CV summary
    if not args.shap_only:
        plot_learning_curves(
            cv_mode=args.use_cv,
            num_epochs=cfg.TRANSYNERGY_PARAMS["num_epochs"],
            cv_folds=cfg.TRANSYNERGY_PARAMS["cv_folds"]
        )
        if args.use_cv:
            summarize_cv_results()

    # 2) Load data & model
    X_all, y_all, feature_names = load_data()
    model = load_model(args.weights, device="cpu")

    # 3) XAI
    print(f"\n[XAI] Running {args.xai}…\n")
    if args.xai in ('shap','all'):
        shap_feature_importance(topk=cfg.TRANSYNERGY_PARAMS.get("topk",20))
    if args.xai in ('lime','all'):
        lime_feature_importance(model, X_all, feature_names,
                                explain_size=cfg.TRANSYNERGY_PARAMS.get("lime_n",10),
                                topk=cfg.TRANSYNERGY_PARAMS.get("topk",20))
    if args.xai in ('anchor','all'):
        anchor_feature_importance(model, X_all, feature_names,
                                  explain_size=cfg.TRANSYNERGY_PARAMS.get("anchor_n",10))

    print("\n[POST] All outputs (CSVs, PNGs, HTMLs) have been generated.\n")