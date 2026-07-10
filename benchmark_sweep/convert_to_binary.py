"""
Relabels dataset_sweep.csv from 3-tier to 2-tier binary system.

Changes:
  - Drops tier2_response, tier2_tokens, tier2_correct (qwen3p7-plus data)
  - Relabels: label = "tier2" (was qwen3p7-plus) -> "tier2" (now means glm-5p2)
  - Renames: old tier3_* columns -> tier2_* (glm-5p2 becomes the new Tier 2)
  - label_encoded: old 0=tier1,1=tier2,2=tier3 -> new 0=tier1,1=tier2
  - All-wrong rows stay as tier2 (escalate fallback)

Result: clean binary dataset ready for router retraining.
"""

import csv
import pandas as pd

INPUT  = "data_builder/dataset_sweep.csv"
OUTPUT = "data_builder/dataset_sweep.csv"  # overwrite in place

print("[*] Loading dataset...")
df = pd.read_csv(INPUT)
print(f"    {len(df)} rows, {len(df.columns)} columns")

print("\n[*] Old label distribution:")
print(df['label'].value_counts().to_string())

# Step 1: Rename tier3_* -> tier2_* (glm-5p2 becomes the new Tier 2)
rename_map = {
    "tier3_response": "tier2_response",
    "tier3_tokens":   "tier2_tokens",
    "tier3_correct":  "tier2_correct",
}
# First drop old tier2 columns (qwen3p7-plus — no longer needed)
cols_to_drop = [c for c in ["tier2_response", "tier2_tokens", "tier2_correct"] if c in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"\n[*] Dropped old tier2 (qwen3p7-plus) columns: {cols_to_drop}")

# Rename tier3 -> tier2
existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
df = df.rename(columns=existing_renames)
print(f"[*] Renamed tier3_* -> tier2_*: {list(existing_renames.keys())}")

# Step 2: Relabel — binary system
# Old: tier1=0, tier2=1 (qwen), tier3=2 (glm)
# New: tier1=0, tier2=1 (glm)
# Any prompt not answered by tier1 -> tier2
def relabel(row):
    t1 = str(row.get("tier1_correct", "0")).strip()
    if t1 == "1":
        return "tier1", 0
    else:
        return "tier2", 1

labels, encoded = zip(*df.apply(relabel, axis=1))
df["label"]         = list(labels)
df["label_encoded"] = list(encoded)

print("\n[*] New label distribution (binary):")
print(df["label"].value_counts().to_string())

# Step 3: Save
df.to_csv(OUTPUT, index=False)
print(f"\n[OK] Saved binary dataset -> {OUTPUT}")
print(f"     Shape: {df.shape}")
print(f"     Columns: {list(df.columns)}")
