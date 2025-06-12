# src/models/train_transsynergy.py

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import TranSynergyDataLoader
from utils.metrics import all_metrics
from models.transynergy_net import TranSynergyNet
from utils.config import cfg

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc="  Training", leave=False)
    for batch_idx, (X_batch, y_batch) in enumerate(loop):
        if batch_idx == 0:
            # 调试信息：检查第一批数据是否有 NaN 或极端值
            import torch
            xb = X_batch.numpy()
            yb = y_batch.numpy()
            print(f"[DEBUG] First batch X statistics: min={xb.min():.2f}, max={xb.max():.2f}, mean={xb.mean():.2f}")
            print(f"[DEBUG] First batch y statistics: min={yb.min():.2f}, max={yb.max():.2f}, mean={yb.mean():.2f}")
            print(f"[DEBUG] Any NaN in X_batch? {np.isnan(xb).any()}, Any Inf in X_batch? {np.isinf(xb).any()}")
            print(f"[DEBUG] Any NaN in y_batch? {np.isnan(yb).any()}, Any Inf in y_batch? {np.isinf(yb).any()}")

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        loop.set_postfix(train_loss=(total_loss / len(loader.dataset)))

    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_trues = []
    total_loss = 0.0

    loop = tqdm(loader, desc="  Evaluating", leave=False)
    with torch.no_grad():
        for X_batch, y_batch in loop:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            all_preds.append(y_pred.cpu().numpy().reshape(-1))
            all_trues.append(y_batch.cpu().numpy().reshape(-1))

            loop.set_postfix(val_loss=(total_loss / len(loader.dataset)))

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    avg_loss = total_loss / len(loader.dataset)
    metrics = all_metrics(trues, preds)
    metrics["loss"] = avg_loss

    return metrics


def run_training():
    params = cfg.TRANSYNERGY_PARAMS
    x_path = params["x_path"]
    y_path = params["y_path"]
    batch_size = params["batch_size"]
    val_ratio = params["val_ratio"]
    test_ratio = params["test_ratio"]
    num_epochs  = params["num_epochs"]
    lr          = params["lr"]
    weight_decay= params["weight_decay"]
    device      = params["device"]  # None means auto-detect
    use_cv      = params["use_cv"]
    cv_folds    = params["cv_folds"]

    if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== Training TransSynergy (use_cv={use_cv}) on device: {device} =====")

    results_list = []

    # 1) Loading data
    data_loader = TranSynergyDataLoader()
    # 2) Choose whether to use cross-validation
    if use_cv:
        folds = data_loader.load_reproduced_data_cv()
        all_fold_results = []
        for fold_idx, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
            print(f"\n----- Fold {fold_idx}/{cv_folds} -----")
            model = TranSynergyNet().to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_mse = float("inf")
            os.makedirs("weights", exist_ok=True)
            best_path = os.path.join("weights", f"best_fold{fold_idx}.pth")
            # Record  losses for plotting later  
            train_losses = []
            val_losses   = []

            for epoch in range(1, num_epochs + 1):
                print(f"\nEpoch {epoch}/{num_epochs}")
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                print(f"→ Train Loss: {train_loss:.2f}")
                train_losses.append(train_loss)


                val_results = evaluate(model, val_loader, criterion, device)
                print("→ Val  | MSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}, Pearson: {:.2f}, Spearman: {:.2f}"
                      .format(val_results["mse"], val_results["mae"], val_results["r2"],
                              val_results["pearson"], val_results["spearman"]))
                val_losses.append(val_results["loss"])

                if val_results["mse"] < best_val_mse:
                    best_val_mse = val_results["mse"]
                    torch.save(model.state_dict(), best_path)
                    print("→ Saved best weights.")

            # 在测试集上评估
            model.load_state_dict(torch.load(best_path))
            test_results = evaluate(model, test_loader, criterion, device)
            print("→ Test | MSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}, Pearson: {:.2f}, Spearman: {:.2f}"
                  .format(test_results["mse"], test_results["mae"], test_results["r2"],
                          test_results["pearson"], test_results["spearman"]))
            
            fold_record = {
                "fold": fold_idx,
                "test_mse": test_results["mse"],
                "test_mae": test_results["mae"],
                "test_r2": test_results["r2"],
                "test_pearson": test_results["pearson"],
                "test_spearman": test_results["spearman"]
            }

            all_fold_results.append(fold_record)
            #
            plt.figure(figsize=(6,4))
            epochs_range = list(range(1, num_epochs+1))
            plt.plot(epochs_range, train_losses, label="Train MSE")
            plt.plot(epochs_range, val_losses,   label="Val MSE")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title(f"Fold {fold_idx} Learning Curve")
            plt.legend()
            plt.grid(True)
            curve_path = f"results/learning_curve_fold{fold_idx}.png"
            plt.tight_layout()
            plt.savefig(curve_path, dpi=200)
            plt.close()
            print(f"[NEW] Saved learning curve for fold {fold_idx} → {curve_path}")
            

        # 汇总 CV 结果
        df_cv = pd.DataFrame(all_fold_results)
        df_cv["model"] = "TransSynergy"
        df_cv["use_cv"] = True
        df_cv.to_csv("results/transynergy_cv_results.csv", index=False)
        print("[SAVE] Saved 5-Fold CV results → transynergy_cv_results.csv")

        # print cv summary
        means = df_cv.mean(numeric_only=True)
        stds  = df_cv.std(numeric_only=True)
        print("\n===== 5-Fold CV Summary =====")
        print(means.to_string(name="mean", float_format="%.4f"))
        print(stds.to_string(name="std",  float_format="%.4f"))

        # ========== [NEW] ==========
        # 绘制 CV 后 Test 指标的均值±标准差柱状图
        metrics_names = ["test_mse", "test_mae", "test_r2", "test_pearson", "test_spearman"]
        means_vals = [means[m] for m in metrics_names]
        stds_vals  = [stds[m]  for m in metrics_names]

        plt.figure(figsize=(8,4))
        sns.set_style("whitegrid")
        x_pos = np.arange(len(metrics_names))
        plt.bar(x_pos, means_vals, yerr=stds_vals, capsize=5, alpha=0.7)
        plt.xticks(x_pos, ["MSE", "MAE", "R2", "Pearson", "Spearman"])
        plt.ylabel("Value")
        plt.title("TranSynergy 5-Fold CV Results (mean ± std)")
        plt.tight_layout()
        plt.savefig("results/transynergy_cv_summary_bar.png", dpi=200)
        plt.close()
        print("[NEW] Saved CV summary bar plot → transynergy_cv_summary_bar.png")

        return df_cv

    else:
        # 单次 80/10/10 划分
        train_loader, val_loader, test_loader = data_loader.load_reproduced_data()

        model = TranSynergyNet().to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_mse = float("inf")
        os.makedirs("weights", exist_ok=True)
        best_path = "weights/model_weights.pth"
        train_losses = []
        val_losses   = []

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"→ Train Loss: {train_loss:.2f}")
            train_losses.append(train_loss)

            val_results = evaluate(model, val_loader, criterion, device)
            print("→ Val  | MSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}, Pearson: {:.2f}, Spearman: {:.2f}"
                  .format(val_results["mse"], val_results["mae"], val_results["r2"],
                          val_results["pearson"], val_results["spearman"]))
            val_losses.append(val_results["loss"])

            if val_results["mse"] < best_val_mse:
                best_val_mse = val_results["mse"]
                torch.save(model.state_dict(), best_path)
                print("→ Saved best weights.")

        # 测试集评估
        model.load_state_dict(torch.load(best_path))
        test_results = evaluate(model, test_loader, criterion, device)
        print("→ Test | MSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}, Pearson: {:.2f}, Spearman: {:.2f}"
              .format(test_results["mse"], test_results["mae"], test_results["r2"],
                      test_results["pearson"], test_results["spearman"]))
        
        single_record = {
            "model": "TransSynergy",
            "use_cv": False,
            "test_mse": test_results["mse"],
            "test_mae": test_results["mae"],
            "test_r2": test_results["r2"],
            "test_pearson": test_results["pearson"],
            "test_spearman": test_results["spearman"]
        }
        df_single = pd.DataFrame([single_record])
        df_single.to_csv("results/transynergy_final_results.csv", index=False)
        print("[NEW] Saved final single-run results → transynergy_final_results.csv")

        # 2) 绘制单次模式的 Test 指标柱状图
        metrics_names = ["test_mse", "test_mae", "test_r2", "test_pearson", "test_spearman"]
        values = [single_record[m] for m in metrics_names]

        plt.figure(figsize=(6,4))
        sns.set_style("whitegrid")
        x_pos = np.arange(len(metrics_names))
        plt.bar(x_pos, values, alpha=0.7)
        plt.xticks(x_pos, ["MSE", "MAE", "R2", "Pearson", "Spearman"])
        plt.ylabel("Value")
        plt.title("TranSynergy Final Test Metrics")
        plt.tight_layout()
        plt.savefig("results/transynergy_final_metrics_bar.png", dpi=200)
        plt.close()
        print("[NEW] Saved final metrics bar plot → transynergy_final_metrics_bar.png")
        # ========== [END NEW] ==========

        # 可以返回一个 DataFrame 或字典，供上层调用者使用（可选）
        return df_single