import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from .config import cfg
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

class SynergyDataLoader:
    def __init__(self):
        print("\n[DATA] Starting to load preprocessed data...")
        self.X = self._load_npy(cfg.X_PATH)
        self.y = self._load_pkl(cfg.Y_PATH)
        print(f"[DATA] X.shape = {self.X.shape},  y.shape = ({len(self.y)},)\n")

        gene_df        = pd.read_csv(cfg.GENE_PATH)
        self.gene_list = gene_df["symbol"].tolist()
        print(f"[DATA] Loaded {len(self.gene_list)} gene names for feature naming")

        expr_names = [f"expr_{g}"   for g in self.gene_list]
        netA_names = [f"drugA_{g}"  for g in self.gene_list]
        netB_names = [f"drugB_{g}"  for g in self.gene_list]
        dep_names  = [f"dep_{g}"    for g in self.gene_list]
        self.feature_names = expr_names + netA_names + netB_names + dep_names
        print(f"[DATA] Built {len(self.feature_names)} feature names\n")


        self.groups, self.folds = self.get_groups()
        self._validate_data()

        print("[DATA] Data loading complete!")
        print(f" - Total samples: {len(self.y)}")
        print(f" - Feature dimension: {self.X.shape[1]}\n")
        print(f" - Feature blocks: 4 x {len(self.gene_list)} = {len(self.feature_names)}")
        print(f" - Fold cols:      {len(self.folds)}\n")

    def _load_npy(self, path):
        print(f"[DATA] Loading feature matrix: {path}")
        return np.load(path)

    def _load_pkl(self, path):
        print(f"[DATA] Loading label file: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f).ravel()
    def _validate_data(self):
        """Validate data consistency"""
        if len(self.X) != len(self.y):
            raise ValueError(
                f"Data mismatch! Number of X samples: {len(self.X)}, number of y samples: {len(self.y)}"
            )
    def get_groups(self):
        """Generate grouping information from metadata"""
        print("\n[DATA] Generating grouping strategy from metadata...")
        meta = pd.read_csv(cfg.SYNERGY_SCORE_PATH)
        drop_cols = [c for c in meta.columns if c.lower().startswith("unnamed")]
        if drop_cols:
            print(f"[DATA]  Dropping columns: {drop_cols}")
            meta = meta.drop(columns=drop_cols)

        n_meta = len(meta)
        n_samples = len(self.y)
        if n_samples % n_meta == 0:
            reps =n_samples // n_meta
            meta = pd.concat([meta] * reps, ignore_index=True)
            print(f"[DATA]  Repeated metadata * {reps} → {len(meta)} rows")
        else:
            raise ValueError(
                f"[ERROR] Cannot align metadata ({n_meta} rows) with "
                f"samples ({n_samples}); please check preprocessing."
            )
        group_tags = {
             'drug_combo': meta.apply(
                 lambda r: '__'.join(sorted([r['drug_a_name'], r['drug_b_name']])), axis=1
             ),
             'cell_line':  meta['cell_line'],
             'drug':       meta['drug_a_name']
         }
         
        fold_ids = {}
        for key, col in [
            ('drug_combo', 'fold'),
            ('cell_line',  'cl_fold'),
            ('drug',       'drug_fold'),
            #('new_drug',   'new_drug_fold'),
            ('random',     'random_fold'),
        ]:
            # Coerce non-numeric to NaN
            vals = pd.to_numeric(meta[col], errors='coerce')
            # Fill NaN with -1, then convert to int
            vals = vals.fillna(-1).astype(int)
            fold_ids[key] = vals
            print(f"[DATA]  - {key}: unique={vals.nunique()}, NaN→-1 count={(vals==-1).sum()}")
            
        # print info
        print(f" - Unique drug_combo folds: {fold_ids['drug_combo'].nunique()}")
        print(f" - Unique cell_line folds:  {fold_ids['cell_line'].nunique()}")
        print(f" - Unique drug folds:       {fold_ids['drug'].nunique()}")
 
        # return two dicts
        return group_tags, fold_ids
    

class TranSynergyDataLoader:
    def __init__(self):
        print("\n[DATA] Starting to load preprocessed data(reproduced by myself)...")
        X_raw = self._load_npy(cfg.X_REPRODUCED_PATH)  #  X (n_samples, n_features)
        y_raw = self._load_pkl(cfg.Y_REPRODUCED_PATH)  #  y (n_samples,)

        print(f"[DATA] Raw X.shape = {X_raw.shape}, Raw y.shape = {y_raw.shape}")

        # ================================================
        # Step A: Standardize features (zero mean, unit variance)
        # ================================================
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_raw.astype(np.float32))
        # Replace NaN, posinf, neginf with 0.0
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        self.X = X_norm
        self.y = y_raw.astype(np.float32)

        print(f"[DATA] Normalized X: mean ~ {np.mean(self.X):.2f}, std ~ {np.std(self.X):.2f}")
        print(f"[DATA] X.shape = {self.X.shape}, y.shape = ({len(self.y)},)\n")

        gene_df        = pd.read_csv(cfg.GENE_PATH)
        self.gene_list = gene_df["symbol"].tolist()
        n_genes = len(self.gene_list)
        print(f"[DATA] Loaded {n_genes} gene names for feature naming")
        rwrA_names = [f"drugA_{g}" for g in self.gene_list]  
        rwrB_names = [f"drugB_{g}" for g in self.gene_list]
        dep_names  = [f"dep_{g}"   for g in self.gene_list]
        expr_names = [f"expr_{g}"  for g in self.gene_list]
        feature_blocks = (
            rwrA_names + 
            rwrB_names + 
            dep_names + 
            expr_names
        )

        
        meta_names = ["fold", "cl_fold", "drug_fold", "new_drug_fold"]
        self.feature_names = feature_blocks + meta_names

        # Debug print a few segments to confirm the start and end positions of each block:
        n = len(self.gene_list)  # 2401
        print("→ RWR_A  first 5:", self.feature_names[0:5])
        print(f"→ RWR_A  last 5:  ", self.feature_names[n-5:n])
        print(f"→ RWR_B  first 5:", self.feature_names[n : n+5])
        print(f"→ DEP    first 5:", self.feature_names[2*n : 2*n+5])
        print(f"→ EXPR   first 5:", self.feature_names[3*n : 3*n+5])
        print("→ META cols:     ", self.feature_names[-4:])

        # print(f"[DATA] Built {len(self.feature_names)} feature names\n")
        #
        # print(self.feature_names[:5],       self.feature_names[2400:2405])
        # print(self.feature_names[4800:4805], self.feature_names[7200:7205])
        # print(self.feature_names[-4:])



        self.batch_size = cfg.TRANSYNERGY_PARAMS['batch_size']
        self.val_ratio = cfg.TRANSYNERGY_PARAMS['val_ratio']
        self.test_ratio = cfg.TRANSYNERGY_PARAMS['test_ratio']
        self.seed = cfg.TRANSYNERGY_PARAMS['seed']
        self.cv_folds = cfg.TRANSYNERGY_PARAMS['cv_folds']
    
    def _load_npy(self, path):
        print(f"[DATA] Loading feature matrix: {path}")
        return np.load(path)

    def _load_pkl(self, path):
        print(f"[DATA] Loading label file: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f).ravel()
    def _validate_data(self):
        """Validate data consistency"""
        if len(self.X) != len(self.y):
            raise ValueError(
                f"Data mismatch! Number of X samples: {len(self.X)}, number of y samples: {len(self.y)}"
            )
    
    def load_reproduced_data(self):
        """
        Loads data from X_reproduce.npy、y_reproduce.pkl, splits it into 3 dataloader:train, validation, and test sets,
        """
        # 1. Loading NumPy arrays 
        X = self.X  # shape：(n_samples, n_features)
        y = self.y
        y = y.reshape(-1)  # Ensure y is a 1D array

        # 2. Convert to torch.Tensor and wrap in TensorDataset
        import torch
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        )

        # 3. Randomly split into train / val / test
        n_total = len(dataset)
        n_val = int(self.val_ratio * n_total)
        n_test = int(self.test_ratio * n_total)
        n_train = n_total - n_val - n_test

        # Ensure reproducible splits
        train_ds, val_ds, test_ds = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # 4. DataLoader
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    
    def load_reproduced_data_cv(self):
        """
        Load data from X_reproduce.npy and y_reproduce.pkl, split it into K-Fold cross-validation sets,
        and return a list of tuples (train_loader, val_loader, test_loader) for each fold.
        """
        X = self.X
        y = self.y
        y = y.reshape(-1)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        )

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        folds = []
        all_indices = np.arange(len(dataset))

        for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(all_indices), start=1):
            # train_val_idx: remaining data，test_idx: current fold test set
            X_train_val = X[train_val_idx]
            y_train_val = y[train_val_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            # From train_val, further sample 10% for validation, the rest for training.
            n_train_val = len(train_val_idx)
            n_val = int(0.1 * n_train_val)
            n_train = n_train_val - n_val

            # random smash-up train_val_idx
            np.random.seed(self.seed)
            perm = np.random.permutation(n_train_val)
            train_sub_idx = train_val_idx[perm[:n_train]]
            val_sub_idx = train_val_idx[perm[n_train:]]

            # build PyTorch Dataset
            train_ds = TensorDataset(
                torch.tensor(X[train_sub_idx], dtype=torch.float32),
                torch.tensor(y[train_sub_idx], dtype=torch.float32).unsqueeze(1)
            )
            val_ds = TensorDataset(
                torch.tensor(X[val_sub_idx], dtype=torch.float32),
                torch.tensor(y[val_sub_idx], dtype=torch.float32).unsqueeze(1)
            )
            test_ds = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            )

            # DataLoader
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

            folds.append((train_loader, val_loader, test_loader))

        return folds


