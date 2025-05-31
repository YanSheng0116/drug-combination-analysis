"""
Preprocess data for TranSynergy:

1. Load 2401-gene list
2. Build normalized PPI adjacency
3. Read drug-target matrix + map to gene indices
4. Run RWR to get drug features
5. Load cell-line dependency & expression
6. Load synergy scores & filter valid samples
7. Assemble X (3x2401) & y and save
"""

import os
import re
import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle

def main():
    # Step 1: Load 2401‐gene list
    gene_file = "data/external/genes_2401_df.csv"
    print(f"1) Loading gene list from {gene_file} …")
    gene_df = pd.read_csv(gene_file)  # header=0 by default
    print("   → Columns found in gene file:", gene_df.columns.tolist())
    print("   → Sample rows:\n", gene_df.head(), "\n")

    # pick the right column for Entrez IDs
    if "entrez" in gene_df.columns:
        entrez_col = "entrez"
    else:
        raise KeyError("Couldn't find an 'entrez' or 'EntrezID' column in gene file.")
    
    symbol_col = "symbol" if "symbol" in gene_df.columns else None

    # Convert to integers / strings:    
    gene_df[entrez_col] = gene_df[entrez_col].astype(int)
    gene_df[symbol_col] = gene_df[symbol_col].astype(str)

    # Build:
    genes        = gene_df[entrez_col].tolist()               # [length 2401]
    symbols      = gene_df[symbol_col].tolist()               # [length 2401]
    symbol2entrez = dict(zip(gene_df[symbol_col], gene_df[entrez_col]))  # for step 5

    print(f"   → Loaded {len(genes)} genes.")
    print(f"   → Example Entrez IDs: {genes[:5]}")
    print(f"   → Example symbols:   {symbols[:5]}\n")

    # Step 2: Read PPI edge list and build normalized adjacency matrix
    # without noralization
    # mat = pd.read_csv("data/external/string_network_matrix.csv", index_col=0)
    # genes = list(mat.index)
    # W = sp.csr_matrix(mat.values)
    # # then directly row‐normalize
    # deg = np.array(W.sum(axis=1)).reshape(-1)
    # D_inv = sp.diags(1.0/deg)
    # W_norm = D_inv.dot(W)

    print("  Building normalized PPI adjacency matrix …")
    # --- Load the precomputed dense PPI matrix (with nomalization)---
    ppi_mat_file = "data/external/string_network_matrix.csv"
    print(f"2) Loading dense PPI matrix from {ppi_mat_file} …")
    mat = pd.read_csv(ppi_mat_file, index_col=0)
    print(f"  → Raw matrix shape: {mat.shape}")
    print(f"  → Example (first 5 genes):\n{mat.iloc[:5, :5]}\n")

    mat_full = mat.reindex(index=genes, columns=genes, fill_value=0)
    print(f"  → Reindexed matrix shape: {mat_full.shape}")
    print(f"  → Example (first 5 genes after reindex):\n{mat_full.iloc[:5, :5]}\n")
    isolated_count = (mat_full.sum(axis=1) == 0).sum()
    print(f"  → Found {isolated_count} isolated genes (zero degree) after reindexing.\n")


    # --- Convert to sparse CSR for efficiency ---
    print("Converting to sparse CSR format …")
    W = sp.csr_matrix(mat_full.values)
    print(f"  → CSR shape: {W.shape}, nnz: {W.nnz}\n")

    # --- Row-normalize to get a transition matrix W_norm ---
    print("Row-normalizing adjacency matrix …")
    deg = np.array(W.sum(axis=1)).reshape(-1)      # degree (row sums)

    print(f"  → Degree stats before fix: min={deg.min():.2f}, max={deg.max():.2f}, mean={deg.mean():.2f}")

    # Avoid division by zero for isolated nodes
    isolated = (deg == 0).sum()
    if isolated:
        print(f"  → Warning: {isolated} isolated genes (zero degree); setting degree=1 to keep zeros in those rows")
        deg[deg == 0] = 1.0

    D_inv = sp.diags(1.0 / deg)                    # D⁻¹
    W_norm = D_inv.dot(W)                          # D⁻¹·W

    # Quick sanity check: each non-isolated row should sum to ~1
    row_sums = np.array(W_norm.sum(axis=1)).reshape(-1)
    print(f"  → Post-norm row sum stats: min={row_sums.min():.6f}, max={row_sums.max():.6f}, mean={row_sums.mean():.6f}")
    print("Row normalization complete — now W_norm is ready for RWR or other graph ops.\n")

    # Step 3: Read drug–target matrix and map to gene‐indices
    dtm_file = "data/external/combine_drug_target_matrix.csv"
    print(f"3) Loading drug-target matrix from {dtm_file} …")
    # read as wide table: rows=drugs, cols=EntrezIDs
    dtm = pd.read_csv(dtm_file, index_col=0)
    # make sure columns are ints
    dtm.columns = dtm.columns.astype(int)
    print(f"   → dtm shape: {dtm.shape} (drugs x genes)")
    print(f"   → columns sample: {dtm.columns[:10].tolist()}")
    print(f"   → drug names sample: {dtm.index[:5].tolist()}\n")
    # Build drug → [gene‐index] map
    drug2idxs = {}
    for drug in dtm.index:
        # find which EntrezIDs have a 1
        targets = dtm.columns[ dtm.loc[drug] == 1 ].tolist()
        # map those EntrezIDs to their positions in our 2401‐gene list
        idxs = [genes.index(g) for g in targets if g in genes]
        drug2idxs[drug] = idxs

    print(f"   → Found {len(drug2idxs)} drugs, example: {list(drug2idxs)[:5]}")
    first = list(drug2idxs)[0]
    print(f"   → For drug '{first}', seed indices: {drug2idxs[first][:10]}\n")

    # Step 4: Run RWR for each drug
    def rwr(seed_idx, W_norm, alpha=0.5, tol=1e-6):
        n = W_norm.shape[0]
        seed = np.zeros(n)
        seed[seed_idx] = 1.0 / len(seed_idx)
        r = seed.copy()
        while True:
            r_next = alpha * W_norm.dot(r) + (1 - alpha) * seed
            if np.linalg.norm(r_next - r, 1) < tol:
                break
            r = r_next
        return r

    print("4) Computing RWR features (α=0.5) for each drug …")
    drug_feats = {}
    for drug, idxs in drug2idxs.items():
        drug_feats[drug] = rwr(idxs, W_norm)
    df_drug = pd.DataFrame(drug_feats, index=genes)
    out_drug = "data/new_reproduced/drug_rwr_features_2401.csv"
    df_drug.to_csv(out_drug)
    print(f"   → Saved drug RWR features {df_drug.shape} to {out_drug}\n")

    # --- 5a) Read the 35 real cell‐line names from the expression CSV header ---
    expr_file = "data/external/processed_expression_raw_norm.csv"
    print(f"5a) Loading expression file header from '{expr_file}' to get 35 cell‐line names …")
    expr_header = pd.read_csv(expr_file, nrows=0)    # only read header row
    expr_cells = list(expr_header.columns)           # this yields the 35 cell‐line names
    print(f"   → Expression file columns (35 cell lines): {expr_cells}")
    print(f"   → Number of columns: {len(expr_cells)}\n")
    if len(expr_cells) != 35:
        raise ValueError(f"Expected 35 columns in expression file, but found {len(expr_cells)}.")

    # --- 5b) Verify that these 35 cell‐line names match synergy_score.csv’s cell_line set ---
    syn_file = "data/external/synergy_score.csv"
    print(f"5b) Loading synergy file '{syn_file}' to verify cell‐line set …")
    syn_df = pd.read_csv(syn_file)
    seen = set()
    unique_from_syn = []
    for cl in syn_df["cell_line"]:
        if cl not in seen:
            seen.add(cl)
            unique_from_syn.append(cl)
    print(f"   → synergy_score.csv cell lines (in first-appearance order): {unique_from_syn}")
    print(f"   → Number of unique cell lines in synergy: {len(unique_from_syn)}\n")

    set_expr = set(expr_cells)
    set_syn = set(unique_from_syn)
    missing_in_expr = set_syn - set_expr
    extra_in_expr = set_expr - set_syn
    if missing_in_expr or extra_in_expr:
        print("   → Mismatch between expression columns and synergy cell_line names:")
        print(f"     • In synergy but not in expression: {sorted(list(missing_in_expr))}")
        print(f"     • In expression but not in synergy: {sorted(list(extra_in_expr))}")
        raise ValueError("The 35 cell‐line names in the expression file must exactly match those in synergy_score.csv.")
    else:
        print("   → OK: Expression file columns and synergy cell_line set match perfectly.\n")

    # --- 5c) Load the raw dependency CSV (35 rows × 18408 columns) ---
    dep_file = "data/external/new_gene_dependencies_35.csv"
    print(f"5c) Loading raw dependency matrix from '{dep_file}' …")
    dep_raw = pd.read_csv(dep_file, header=0)
    print(f"   → Original dep_raw shape: {dep_raw.shape} (rows × cols)")
    print(f"   → Original dep_raw.columns[:5]: {list(dep_raw.columns[:5])}\n")

    # If pandas created an "Unnamed: 0" column from a leftover index, drop it
    if "Unnamed: 0" in dep_raw.columns:
        print("   → Dropping column 'Unnamed: 0' (leftover index column).")
        dep_raw = dep_raw.drop(columns=["Unnamed: 0"])
        print(f"   → New dep_raw shape after drop: {dep_raw.shape}")
        print(f"   → Columns now: {list(dep_raw.columns[:5])}\n")

    # --- 5d) Assign the expression file’s column order as dep_raw’s row index ---
    print("5d) Assigning dep_raw.index = expr_cells (so row order = expression column order) …")
    dep_raw.index = expr_cells # type: ignore
    print(f"   → Now dep_raw.index[:5]: {dep_raw.index[:5].tolist()}\n") # type: ignore

    # --- 5e) Parse all "Symbol (Entrez)" column headers to extract pure Entrez IDs ---
    print("5e) Parsing 'Symbol (EntrezID)' from each column name …")
    pattern = re.compile(r"^(?P<sym>.+)\s*\((?P<eid>\d+)\)$")
    old_cols = list(dep_raw.columns)         # e.g. length 18408
    parsed_entrez = []
    parsed_symbol = []
    for col in old_cols:
        m = pattern.match(col.strip())
        if not m:
            raise ValueError(f"Column '{col}' does not match 'Symbol (Entrez)' format.")
        sym = m.group("sym")
        eid = int(m.group("eid"))
        # We store both symbol and eid; later we'll filter to genes ∈ our 2401 list
        parsed_symbol.append(sym)
        parsed_entrez.append(eid)
    parsed_entrez = np.array(parsed_entrez, dtype=int)
    print(f"   → Parsed first 5 symbols: {parsed_symbol[:5]}")
    print(f"   → Parsed first 5 Entrez IDs: {parsed_entrez[:5]}\n")

    # Rename the columns from "Symbol (Entrez)" → pure EntrezID integer
    print("5f) Renaming columns to pure Entrez IDs …")
    col_map = {old: new for old, new in zip(old_cols, parsed_entrez)}
    dep_raw = dep_raw.rename(columns=col_map)
    print(f"   → After renaming, dep_raw.columns[:5]: {list(dep_raw.columns[:5])}\n")

    # --- 5g) Filter columns to keep only those Entrez IDs in our 2401‐gene list ---
    print("5g) Filtering dep_raw to keep only the 2401 genes in 'genes' …")
    dep_filtered = dep_raw.loc[:, dep_raw.columns.isin(genes)]
    print(f"   → Shape after filtering: {dep_filtered.shape} (should be 35 × 2401)")
    if dep_filtered.shape[1] != len(genes):
        missing = set(genes) - set(dep_filtered.columns)
        print(f"   → WARNING: {len(missing)} of the 2401 genes not found in dependency columns!")
        print(f"     Example missing Entrez IDs: {sorted(list(missing))[:10]} …")
        # Add zero‐filled columns for any missing genes
        for m in missing:
            dep_filtered[m] = 0.0
        # Re‐order columns to exactly match the `genes` list
        dep_filtered = dep_filtered.reindex(columns=genes, fill_value=0.0)
        print(f"   → After adding missing, dep_filtered shape: {dep_filtered.shape}\n")
    else:
        # Exactly 2401 matched; reorder columns if necessary
        dep_filtered = dep_filtered.reindex(columns=genes)
        print("   → All 2401 genes present. Columns reindexed to match master gene list.\n")

    # --- 5h) Transpose so that rows = EntrezID (2401), columns = cell lines (35) ---
    print("5h) Transposing dep_filtered to get final 'dep' DataFrame …")
    dep = dep_filtered.transpose()
    print(f"   → Final dep.shape: {dep.shape} (2401 × 35)")
    print(f"   → dep.index[:5] (Entrez IDs): {dep.index[:5].tolist()}")
    print(f"   → dep.columns[:5] (cell lines): {dep.columns[:5].tolist()}\n")

    # === At this point, dep is ready: shape (2401, 35), index = 2401 Entrez IDs, columns = 35 cell lines ===

    # --- Step 5b) Load & align the expression matrix (2401 × 35) ---
    print(f"5i) Loading raw expression matrix from {expr_file}…")
    expr_raw2 = pd.read_csv(expr_file, header=0)
    print(f"   → expr_raw2 shape: {expr_raw2.shape}")
    # Confirm its columns are exactly expr_cells
    assert list(expr_raw2.columns) == expr_cells, \
        "Expression file columns do not match previously read expr_cells."
    print("   → Expression file columns confirmed to match expr_cells.\n")

    # Assign the row index = genes (2401 Entrez IDs)
    print("5j) Assigning expr_raw2.index = genes (2401 Entrez IDs) …")
    expr_raw2.index = genes # type: ignore
    print(f"   → expr_raw2.index[:5]: {expr_raw2.index[:5].tolist()}") # type: ignore
    print(f"   → expr_raw2 shape after reindex: {expr_raw2.shape}\n")

    # The final expression DataFrame `expr`:
    expr = expr_raw2.copy()
    print("   → Final expr.shape: ", expr.shape)
    print("   → expr.index[:5] (Entrez IDs):", expr.index[:5].tolist())
    print("   → expr.columns[:5] (cell lines):", expr.columns[:5].tolist(), "\n")

    print(f" Steps 5 completed: dep (shape: {dep.shape}) and expr (shape: {expr.shape}) are ready.\n")

  


    # Step 6: Load synergy scores and filter
    syn_file = "data/external/synergy_score.csv"
    print(f"6) Loading synergy scores from {syn_file} …")
    syn = pd.read_csv(syn_file)
    print("=== Checking input shapes ===")
    print(f"df_drug shape:  {df_drug.shape}  (should be 2401×462)")
    print(f"dep shape:      {dep.shape}      (should be 2401×35)")
    print(f"expr shape:     {expr.shape}     (should be 2401×35)")
    print(f"syn shape:      {syn.shape}     (shoud be 18552 rows)\n")

    print(f"   → Total records before filter: {len(syn)}")
    
    # For each row, we will generate two samples: (drugA,drugB) and (drugB,drugA)
    # Ensure we have all required columns:
    required_cols = ["drug_a_name","drug_b_name","cell_line","synergy",
                     "fold","cl_fold","drug_fold","new_drug_fold"]
    for col in required_cols:
        if col not in syn.columns:
            raise KeyError(f"synergy_score.csv missing required column '{col}'")
    # check that both drugA and drugB exist in df_drug, and that cell_line exists in expr.
    
    # Generate both forward and reverse combinations
    syn["combined_forward"] = syn["drug_a_name"] + "_" + syn["drug_b_name"]   # forward key
    syn["combined_reverse"] = syn["drug_b_name"] + "_" + syn["drug_a_name"]   # reverse key

    # Keep any row where either forward or reverse exists in df_drug.columns,
    # AND cell_line exists in expr.columns
    valid_mask = (
        (syn["combined_forward"].isin(df_drug.columns) |                         
         syn["combined_reverse"].isin(df_drug.columns)) &                       
        syn["cell_line"].isin(expr.columns)
    )
    syn = syn[valid_mask].reset_index(drop=True)
    print(f"   → Records after basic filter: {len(syn)}\n")



    # Step 7: Assemble full X (37104×9608) and y (37104×1)
    print("7) Assembling full X (n_samples×9608) and y (n_samples×1) …")
    n_orig = len(syn)
    n_samples = n_orig * 2  # each row → two samples (drugA→drugB and drugB→drugA)
    g = len(genes)  # 2401

    # Each sample has 4 channels of 2401 dims + 4 extra dims = 2401*4 + 4 = 9608
    n_features = g * 4 + 4
    print(f"   → Expecting {n_orig} original rows → {n_samples} total samples")
    print(f"   → Each sample dimension: 2401*4 + 4 = {n_features}\n")

    # Pre-allocate
    X = np.zeros((n_samples, n_features), dtype=float)
    y = np.zeros((n_samples, 1), dtype=float)

    idx = 0
    for i, row in syn.iterrows():
        if row["combined_forward"] in df_drug.columns:    
            combo = row["combined_forward"]            
        else:
            combo = row["combined_reverse"]
        cell  = row["cell_line"]
        score = float(row["synergy"])

        # Extract four 2401‐dim channels:
        vecA   = df_drug[combo].values        # (2401,)
        vecB   = df_drug[combo].values        # (2401,)
        dep_vec  = dep[cell].values           # (2401,)
        expr_vec = expr[cell].values          # (2401,)

        # Extract 4 extra metadata dims:
        fold_val       = float(row["fold"])
        cl_fold_val    = float(row["cl_fold"])
        drug_fold_val  = float(row["drug_fold"])
        new_drug_val   = float(row["new_drug_fold"])
        extra_feats = np.array([fold_val, cl_fold_val, drug_fold_val, new_drug_val], dtype=float)

        # 1) Forward order: (drugA→drugB)
        sample_feat = np.concatenate([vecA, vecB, dep_vec, expr_vec, extra_feats])
        X[idx, :] = sample_feat
        y[idx, 0] = score
        idx += 1

        # 2) Reverse order: (drugB→drugA)
        sample_feat_rev = np.concatenate([vecB, vecA, dep_vec, expr_vec, extra_feats])
        X[idx, :] = sample_feat_rev
        y[idx, 0] = score
        idx += 1

        # Print progress every 2000 original rows
        if (i + 1) % 2000 == 0: # type: ignore
            print(f"   → Processed {i+1}/{n_orig} original rows → generated {idx} samples so far") # type: ignore

    # Truncate if any rows were skipped (unlikely):
    if idx != n_samples:
        print(f"Warning: actually generated {idx} samples instead of expected {n_samples}. Truncating.")
        X = X[:idx, :]
        y = y[:idx, :]

    print(f"\n   → Final X.shape: {X.shape}")
    print(f"   → Final y.shape: {y.shape}\n")

    # Save the full X and y
    
    x_out = "data/new_reproduced/X_full_reproduce.npy"
    y_out = "data/new_reproduced/y_full_reproduce.pkl"
    np.save(x_out, X)
    with open(y_out, "wb") as f:
        pickle.dump(y, f)
    print(f"   → Saved full X to {x_out}")
    print(f"   → Saved full y to {y_out}\n")
    print(f"✅ Full reproduction of X (shape: {X.shape}) and y (shape: {y.shape}) complete!\n")


if __name__ == "__main__":
    main()
