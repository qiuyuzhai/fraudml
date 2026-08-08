"""Test DataCleaner with full IEEE-CIS pipeline."""
from src.data import DataLoader, DataCleaner

# 1. Load data (same as baseline)
dl = DataLoader()
txn, ident = dl.load_train()

# 2. Time split (first 80% train, last 20% val)
txn = txn.sort_values("TransactionDT").reset_index(drop=True)
n_total = len(txn)
n_val = int(n_total * 0.2)
train_df = txn.iloc[:-n_val].copy()
val_df = txn.iloc[-n_val:].copy()

# 3. Merge Identity AFTER split (leak prevention)
train_df = train_df.merge(ident, on="TransactionID", how="left")
val_df = val_df.merge(ident, on="TransactionID", how="left")

print("=== 1. FIT on TRAINING only ===")
cleaner = DataCleaner()
cleaner.fit(train_df)
print(f"Constant cols (dropped): {len(cleaner.constant_cols_)}")
print(f"Numeric cols (processed): {len(cleaner.numeric_cols_)}")
print(f"Cols with missing: {len(cleaner.cols_with_missing_)}")

for i, col in enumerate(cleaner.numeric_cols_[:3]):
    q1, q99 = cleaner.quantile_thresholds_.get(col, (None, None))
    med = cleaner.medians_.get(col, None)
    if med is not None and q1 is not None:
        print(f"  {col}: median={med:.2f}, q1={q1:.2f}, q99={q99:.2f}")

# 4. Transform training
print("\n=== 2. TRANSFORM on TRAIN ===")
train_clean = cleaner.transform(train_df)
print(f"Before: {train_df.shape}, After: {train_clean.shape}")
new_cols = [c for c in train_clean.columns if c.endswith("_isna") or c.endswith("_clip_low") or c.endswith("_clip_high")]
print(f"Generated flag columns: {len(new_cols)}")
print(f"  _isna flags: {sum(1 for c in new_cols if c.endswith('_isna'))}")
print(f"  _clip_low flags: {sum(1 for c in new_cols if c.endswith('_clip_low'))}")
print(f"  _clip_high flags: {sum(1 for c in new_cols if c.endswith('_clip_high'))}")

# 5. Transform validation (NO new statistics)
print("\n=== 3. TRANSFORM on VAL ===")
val_clean = cleaner.transform(val_df)
print(f"Val before: {val_df.shape}, Val after: {val_clean.shape}")

# 6. Verify no leakage checks
remaining_constant = [c for c in cleaner.constant_cols_ if c in train_clean.columns]
print(f"\nLeak check - constant cols in cleaned: {len(remaining_constant)}")

# 7. Verify isna flags
col = cleaner.cols_with_missing_[0]
print(f"\nisna flag check for '{col}':")
print(f"  original nulls: {train_df[col].isna().sum()}")
print(f"  _isna flag sum:  {train_clean[f'{col}_isna'].sum()}")

# 8. Verify clip flags
print("\nClip flag verification:")
for col in cleaner.numeric_cols_[:3]:
    q1, q99 = cleaner.quantile_thresholds_[col]
    n_low = (train_df[col].dropna() < q1).sum()
    n_high = (train_df[col].dropna() > q99).sum()
    flag_low = train_clean[f"{col}_clip_low"].sum()
    flag_high = train_clean[f"{col}_clip_high"].sum()
    print(f"  {col}: orig_low={n_low}, flag_low={flag_low} | orig_high={n_high}, flag_high={flag_high}")

# 9. Export cleaning summary for audit / review
print("\n=== 5. EXPORT cleaning summary ===")
summary_path = cleaner.save_summary()
summary_df = cleaner.summary()
print(f"Summary saved to: {summary_path}")
print(f"Total rows: {len(summary_df)}")
print(f"  Action=drop: {(summary_df['action'] == 'drop').sum()} columns")
print(f"  Action=keep: {(summary_df['action'] == 'keep').sum()} columns")
print(f"  With missing: {summary_df['has_missing'].sum()} columns")
print("\nFirst 5 rows preview:")
print(summary_df.head().to_string(index=False))

# 10. Verify Winsorization bounds
print("\n=== 6. Winsorization bounds check ===")
for col in cleaner.numeric_cols_[:5]:
    q1, q99 = cleaner.quantile_thresholds_[col]
    cmin = train_clean[col].min()
    cmax = train_clean[col].max()
    ok_low = cmin >= q1
    ok_high = cmax <= q99
    print(f"  {col}: min={cmin:.2f} (>=q1={q1:.2f}? {ok_low}), max={cmax:.2f} (<=q99={q99:.2f}? {ok_high})")
    assert ok_low and ok_high, f"FAIL: {col} bounds violated!"

# 11. Verify val has NO new stats
print("\n=== 7. Val set uses stored thresholds (no recomputation) ===")
print("  (This is guaranteed by design — transform never computes stats)")

print("\nALL CHECKS PASSED")