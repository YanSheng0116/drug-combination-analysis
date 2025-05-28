import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from .config import cfg

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
            # 强制转数字，非数值变 NaN
            vals = pd.to_numeric(meta[col], errors='coerce')
            # 将 NaN 填成 -1，然后转 int
            vals = vals.fillna(-1).astype(int)
            fold_ids[key] = vals
            print(f"[DATA]  - {key}: unique={vals.nunique()}, NaN→-1 count={(vals==-1).sum()}")
            
        # print info
        print(f" - Unique drug_combo folds: {fold_ids['drug_combo'].nunique()}")
        print(f" - Unique cell_line folds:  {fold_ids['cell_line'].nunique()}")
        print(f" - Unique drug folds:       {fold_ids['drug'].nunique()}")
 
        # return two dicts
        return group_tags, fold_ids