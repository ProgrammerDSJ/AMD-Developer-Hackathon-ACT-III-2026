import pandas as pd
import csv

df = pd.read_csv('data_builder/dataset_sweep.csv', quoting=csv.QUOTE_MINIMAL)

print("=== SUSPICIOUSLY SHORT RESPONSES (< 10 chars) ===")
for col in ['tier1_response', 'tier2_response', 'tier3_response']:
    short = df[df[col].astype(str).str.len() < 10]
    if len(short):
        print(f"\n{col} ({len(short)} rows):")
        for _, r in short.iterrows():
            pid = r['prompt_id']
            src = r['source']
            val = repr(str(r[col]))
            print(f"  {pid} ({src}): {val}")
    else:
        print(f"\n{col}: none short - OK")

print()
print("=== SAMPLE SPOT CHECK (5 random full responses) ===")
import random
random.seed(42)
sample_idx = random.sample(range(len(df)), 5)
for i in sample_idx:
    row = df.iloc[i]
    print(f"\n--- Row {i} | {row['prompt_id']} | {row['source']} ---")
    print(f"  Prompt (first 80 chars): {str(row['prompt'])[:80]!r}")
    print(f"  Tier1 ({len(str(row['tier1_response']))} chars): {str(row['tier1_response'])[:120]!r}")
    print(f"  Tier2 ({len(str(row['tier2_response']))} chars): {str(row['tier2_response'])[:120]!r}")
    print(f"  Tier3 ({len(str(row['tier3_response']))} chars): {str(row['tier3_response'])[:120]!r}")

print()
print("=== NEWLINE INTEGRITY CHECK ===")
# Responses with newlines should be quoted in CSV - verify pandas reads them correctly
for col in ['tier1_response', 'tier2_response', 'tier3_response']:
    has_newline = df[col].astype(str).str.contains('\n', na=False).sum()
    print(f"  {col}: {has_newline} responses contain newlines (multiline - normal for long answers)")

print()
print("=== TOKEN COUNT vs RESPONSE LENGTH CORRELATION ===")
for tier, tok_col, resp_col in [
    ('Tier1', 'tier1_tokens', 'tier1_response'),
    ('Tier2', 'tier2_tokens', 'tier2_response'),
    ('Tier3', 'tier3_tokens', 'tier3_response'),
]:
    tokens = pd.to_numeric(df[tok_col], errors='coerce')
    resp_len = df[resp_col].astype(str).str.len()
    # Rows where we have tokens but extremely short response (suspicious)
    suspicious = df[(tokens > 100) & (resp_len < 20)]
    print(f"  {tier}: {len(suspicious)} rows with >100 tokens but <20 char response")

print()
print("=== FINAL VERDICT ===")
total = len(df)
all_filled = all(
    (df[c].astype(str).str.strip().isin(['', 'nan', 'None', 'NaN'])).sum() == 0
    for c in ['tier1_response', 'tier2_response', 'tier3_response']
)
print(f"Total rows       : {total}")
print(f"All tiers filled : {all_filled}")
print(f"Row count OK     : {total == 779}")
