import pandas as pd

df = pd.read_csv('data_builder/dataset_sweep.csv')

print("=== FILE: data_builder/dataset_sweep.csv ===")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")
print()

print("=== ALL COLUMNS (with type and sample value) ===")
for i, col in enumerate(df.columns):
    dtype = str(df[col].dtype)
    sample = str(df[col].iloc[0])[:45].replace('\n', ' ')
    print(f"  [{i+1:02d}] {col:<35} ({dtype:<8})  e.g. {sample!r}")

print()
print("=== ML FEATURE COLUMNS (13 input features for XGBoost/LightGBM) ===")
feature_cols = [
    'source_task_type_encoded',
    'prompt_length',
    'has_code_block',
    'has_math_symbols',
    'question_type_encoded',
    'num_sentences',
    'avg_word_length',
    'complexity_heuristic',
    'llm_reasoning_depth',
    'llm_ambiguity_score',
    'llm_context_dependency',
    'llm_requires_factual_recall',
    'llm_task_type',
]
for col in feature_cols:
    status = "OK" if col in df.columns else "MISSING"
    print(f"  {col:<35} [{status}]")

print()
print("=== TARGET / LABEL COLUMNS ===")
label_counts = df['label'].value_counts().to_dict()
enc_counts   = df['label_encoded'].astype(str).value_counts().to_dict()
print(f"  label         : {label_counts}")
print(f"  label_encoded : {enc_counts}")

print()
print("=== METADATA / CONTEXT COLUMNS (not fed to ML) ===")
meta_cols = ['prompt_id','source','task_type','domain','difficulty',
             'prompt','reference_answer','question_type']
for col in meta_cols:
    status = "present" if col in df.columns else "missing"
    print(f"  {col:<25} [{status}]")

print()
print("=== SWEEP OUTPUT COLUMNS (not fed to ML) ===")
sweep_cols = ['tier1_response','tier1_tokens','tier1_correct',
              'tier2_response','tier2_tokens','tier2_correct',
              'tier3_response','tier3_tokens','tier3_correct']
for col in sweep_cols:
    status = "present" if col in df.columns else "missing"
    print(f"  {col:<25} [{status}]")

print()
print("=== COMPLETENESS CHECK ===")
print(f"  label filled       : {df['label'].notna().sum()}/{len(df)}")
print(f"  label_encoded      : {df['label_encoded'].notna().sum()}/{len(df)}")
for col in ['tier1_correct','tier2_correct','tier3_correct']:
    n = df[col].astype(str).isin(['0','1']).sum()
    print(f"  {col:<25}: {n}/{len(df)}")
