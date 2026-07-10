from inference_wrapper.feature_extractor import extract_features
from inference_wrapper.router_core import predict

prompts = [
    ("What is 2 + 2?", "simple math"),
    ("Write a Python function that reverses a linked list in-place.", "code"),
    ("Explain the philosophical implications of the Ship of Theseus paradox.", "open-ended"),
    ("If x + 5 = 12, what is x?\nA. 5\nB. 7\nC. 8\nD. 17\nAnswer:", "mcq"),
]

print(f"{'Task':<12} {'tier1_prob':>10} {'tier2_prob':>10}  {'Decision':>8}   Prompt")
print("-"*80)
for p, label in prompts:
    f = extract_features(p)
    tier, t1p, t2p = predict(f)
    arrow = ">> LOCAL" if t1p >= 0.6 else ("-> TIER1" if tier == "tier1" else "-> TIER2")
    print(f"{label:<12}  {t1p:>9.3f}  {t2p:>9.3f}  {arrow:>10}   {p[:55].replace(chr(10),' ')}")
