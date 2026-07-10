import pandas as pd
import numpy as np

df = pd.read_csv('data_builder/dataset_sweep.csv')

print("=" * 60)
print(" DATA DISTRIBUTION ANALYSIS")
print("=" * 60)

print("\n--- 1. OVERALL LABEL DISTRIBUTION ---")
label_counts = df['label'].value_counts()
for label, count in label_counts.items():
    bar = "#" * int(count / 779 * 50)
    print(f"  {label:<8} : {count:>4} rows ({count/779*100:.1f}%)  {bar}")

print("\n--- 2. LABEL DISTRIBUTION BY SOURCE ---")
ct = pd.crosstab(df['source'], df['label'])
ct['total'] = ct.sum(axis=1)
for col in ['tier1','tier2','tier3']:
    if col not in ct.columns:
        ct[col] = 0
ct = ct[['tier1','tier2','tier3','total']]
print(f"  {'source':<12} {'tier1':>7} {'tier2':>7} {'tier3':>7} {'total':>7}")
print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for src, row in ct.iterrows():
    t1_pct = row['tier1']/row['total']*100
    t2_pct = row['tier2']/row['total']*100
    t3_pct = row['tier3']/row['total']*100
    print(f"  {src:<12} {int(row['tier1']):>4}({t1_pct:.0f}%) {int(row['tier2']):>4}({t2_pct:.0f}%) {int(row['tier3']):>4}({t3_pct:.0f}%) {int(row['total']):>7}")

print("\n--- 3. WHAT MAKES A tier2 PROMPT? ---")
t2 = df[df['label'] == 'tier2']
t1 = df[df['label'] == 'tier1']
t3 = df[df['label'] == 'tier3']
print(f"  Total tier2 rows: {len(t2)}")
print(f"  Source breakdown:")
print(t2['source'].value_counts().to_string())
print(f"\n  Task type breakdown:")
print(t2['task_type'].value_counts().to_string())
print(f"\n  Avg feature values vs other tiers:")
num_cols = ['prompt_length','complexity_heuristic','llm_reasoning_depth',
            'llm_ambiguity_score','has_math_symbols','has_code_block']
print(f"  {'Feature':<30} {'tier1':>8} {'tier2':>8} {'tier3':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
for col in num_cols:
    m1 = t1[col].mean()
    m2 = t2[col].mean()
    m3 = t3[col].mean()
    print(f"  {col:<30} {m1:>8.3f} {m2:>8.3f} {m3:>8.3f}")

print("\n--- 4. CLASS IMBALANCE METRICS ---")
total = len(df)
t1_n, t2_n, t3_n = len(t1), len(t2), len(t3)
print(f"  tier1 : {t1_n} ({t1_n/total*100:.1f}%)  -- majority class")
print(f"  tier2 : {t2_n} ({t2_n/total*100:.1f}%)  -- severe minority class")
print(f"  tier3 : {t3_n} ({t3_n/total*100:.1f}%)")
print(f"  Imbalance ratio (tier1:tier2): {t1_n/t2_n:.1f}:1  <-- very skewed")
print(f"  Imbalance ratio (tier3:tier2): {t3_n/t2_n:.1f}:1")

print("\n--- 5. IMPROVEMENT STRATEGIES ---")
print("""
  A. Address class imbalance (tier2):
     1. class_weight='balanced' or scale_pos_weight in XGBoost
     2. SMOTE (Synthetic Minority Over-sampling) on tier2 rows
     3. Collect more tier2 data (prompts where tier1 fails, tier2 succeeds)

  B. Feature engineering:
     1. Drop llm_context_dependency (0% importance -- dead feature)
     2. Add interaction features: prompt_length * llm_reasoning_depth
     3. Add source-specific difficulty proxy

  C. Model tuning:
     1. Tune class_weight / sample_weight for tier2
     2. Grid search hyperparameters (n_estimators, max_depth, min_child_weight)
     3. Try treating as ordinal (tier1 < tier2 < tier3) with ordinal regression

  D. Label strategy:
     1. Merge tier2 into tier3 (binary: tier1 vs tier2/tier3)
     2. Train binary router first, then a tier2 vs tier3 sub-classifier
""")

print("--- 6. CONFUSION MATRIX ANALYSIS (from training output) ---")
print("""
  Actual tier1 -> Predicted tier1: 84/97  (87% recall)  GOOD
  Actual tier1 -> Predicted tier3: 12/97  -- model over-escalates 12% of easy prompts
  Actual tier3 -> Predicted tier1: 11/51  -- model under-escalates 22% of hard prompts
  Actual tier2 -> Predicted tier2:  1/8   -- nearly blind to tier2 (12% recall)
  Actual tier2 -> Predicted tier1:  5/8   -- tier2 gets misclassified as tier1 mostly
""")
