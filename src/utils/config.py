from pathlib import Path

class Config:
    # root directory and data directories
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    EXTERNAL_DIR = DATA_DIR / "external"
    PROCESSED_DIR = DATA_DIR / "processed"
    NEW_REPRODUCED_DIR = DATA_DIR / "new_reproduced"

    # data path
    X_PATH = PROCESSED_DIR / "X.npy"
    Y_PATH = PROCESSED_DIR / "y.pkl"
    SYNERGY_SCORE_PATH = EXTERNAL_DIR / "synergy_score.csv"
    GENE_PATH = EXTERNAL_DIR / "genes_2401_df.csv"
    X_REPRODUCED_PATH = NEW_REPRODUCED_DIR / "X_full_reproduce.npy"
    Y_REPRODUCED_PATH = NEW_REPRODUCED_DIR / "y_full_reproduce.pkl"

    # model parameters
    RF_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'n_jobs': -1
    }
    
    TRANSYNERGY_PARAMS = {
        "x_path": X_REPRODUCED_PATH,
        "y_path": Y_REPRODUCED_PATH,
        "input_dim": 9608,
        "hidden_dim": 2048,
        "dropout": 0.5,
        "batch_size": 128,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "num_epochs": 50,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "device": None,
        "use_cv": False,
        "cv_folds": 5,
        "seed": 42
    }

    RANDOM_SEED = 42
    PCA_COMPONENTS = 512

cfg = Config()

print("[CONFIG] Configuration file loaded successfully, key path validation:")
print(f" - Synergy data path: {cfg.SYNERGY_SCORE_PATH.exists()}")
print(f" - Gene names path: {cfg.GENE_PATH.exists()}")
print(f" - Feature path: {cfg.X_PATH.exists()}")
print(f" - Label path: {cfg.Y_PATH.exists()}")
print(f" - Reproduced feature path: {cfg.X_REPRODUCED_PATH.exists()}")
print(f" - Reproduced label path: {cfg.Y_REPRODUCED_PATH.exists()}")
print("[CONFIG] Configuration file loaded successfully!\n")