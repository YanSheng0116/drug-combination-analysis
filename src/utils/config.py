from pathlib import Path

class Config:
    # root directory and data directories
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    EXTERNAL_DIR = DATA_DIR / "external"
    PROCESSED_DIR = DATA_DIR / "processed"

    # data path
    X_PATH = PROCESSED_DIR / "X.npy"
    Y_PATH = PROCESSED_DIR / "y.pkl"
    SYNERGY_SCORE_PATH = EXTERNAL_DIR / "synergy_score.csv"
    GENE_PATH = EXTERNAL_DIR / "genes_2401_df.csv"    

    # model parameters
    RF_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'n_jobs': -1
    }
    RANDOM_SEED = 42
    PCA_COMPONENTS = 512

cfg = Config()

print("[CONFIG] Configuration file loaded successfully, key path validation:")
print(f" - Synergy data path: {cfg.SYNERGY_SCORE_PATH.exists()}")
print(f" - Gene names path: {cfg.GENE_PATH.exists()}")
print(f" - Feature path: {cfg.X_PATH.exists()}")
print(f" - Label path: {cfg.Y_PATH.exists()}\n")