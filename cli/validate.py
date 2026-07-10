import sys
sys.path.insert(0, 'D:/Hackathon/AMD Developer Hackathon ACT II 2026')

from calibration.run_calibration import load_config
from inference_wrapper.feature_extractor import extract_features
from inference_wrapper.router_core import predict

cfg    = load_config()
models = cfg.get("models", {})
active = cfg.get("active_model")
thresh = models.get(active, {}).get("local_threshold", 1.0) if active else 1.0

print(f"Config loaded. Active model: {active}")
print(f"Calibrated models: {list(models.keys())}")
if active:
    m = models[active]
    print(f"Cal acc={m['calibration_acc']*100:.1f}%  threshold={thresh}")
print()

tests = [
    ("What is 2 + 2?",                                             "arithmetic"),
    ("Write a Python quicksort implementation.",                   "code"),
    ("Capital of France? A.Berlin B.Paris C.Rome D.Madrid Ans:", "mcq"),
    ("Analyze ethical implications of autonomous weapon systems.", "complex"),
]
print(f"{'Prompt':<52}  {'task':<10}  {'t1_prob':>8}  {'Decision':>10}")
print("-" * 88)
for p, label in tests:
    f = extract_features(p)
    tier, t1p, t2p = predict(f)
    use_local = active and t1p >= thresh
    dest = "LOCAL" if use_local else ("TIER1" if tier == "tier1" else "TIER2")
    print(f"{p[:52]:<52}  {f['_task_type']:<10}  {t1p:>8.3f}  {dest:>10}")
