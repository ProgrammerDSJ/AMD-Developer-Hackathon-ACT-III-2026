session2_log = """

---

## Session 2 -- July 11, 2026 (~02:40-04:00 IST)

### What Was Built

#### 3-Layer Routing Architecture
```
Prompt -> Feature Extraction (15 features, <5ms)
       -> Router (LightGBM, <1ms) -> tier1_prob score
       -> Local Gate (threshold check)
              |                          |
     tier1_prob >= threshold        tier1_prob < threshold
              |                          |
         Ollama local (0 tokens)    Fireworks remote (tier1 or tier2)
```

#### Auto-Calibration (per-device, per-model)
- 100 stratified calibration prompts, deterministic scoring, zero API tokens
- Maps local model accuracy to routing threshold via calibration curve
- Config stored at ~/.hybridrouter/config.json, reused across sessions
- Recalibrate anytime: `python cli/main.py --recalibrate [model]`

All 3 local models calibrated:

| Model | Cal Acc | Threshold | MMLU | ARC | GSM8K |
|---|---|---|---|---|---|
| qwen2.5:0.5b | 28% | 0.95 | 20% | 32% | 4% |
| smollm2:135m | 21% | 0.95 | 12% | 16% | 0% |
| smollm2:360m | 34% | 0.95 | 44% | 32% | 0% |

> With better local models (llama3.2:3b, qwen3:8b), threshold drops to 0.40-0.52,
> routing 48-60% of prompts locally at near-zero token cost.

#### Scorer Bug Fix (critical)
- **Root cause:** MCQ scorer grabbed first letter in text. Models doing step-by-step
  analysis always mention option A first, so correct B/C/D answers were marked wrong.
- **Fix:** Priority extraction -- explicit markers first, then last letter in tail, then first
- **Validated:** `calibration/test_scorer.py` -- 8/8 tests passing

#### Unified CLI (cli/main.py)
```
python cli/main.py                              # Interactive mode
python cli/main.py "prompt"                     # Single-shot
python cli/main.py --demo                       # 6-prompt demo
python cli/main.py --stats                      # Token summary
python cli/main.py --recalibrate                # Recalibrate (pick model from list)
python cli/main.py --recalibrate qwen2.5:0.5b  # Recalibrate specific model
```
Interactive commands: `recalibrate [model]`, `switch`, `stats`, `exit`

### Files Created/Modified This Session
```
inference_wrapper/feature_extractor.py   NEW
inference_wrapper/local_client.py        NEW
inference_wrapper/fireworks_client.py    NEW
inference_wrapper/router_core.py         NEW
calibration/extract_calibration_set.py  NEW
calibration/calibration_prompts.jsonl   NEW (100 prompts)
calibration/run_calibration.py          UPDATED (all models, improved scorer)
calibration/diagnose_scorer.py          NEW (diagnostic tool)
calibration/test_scorer.py              NEW (8 unit tests)
cli/main.py                             UPDATED (unified app, recalibrate support)
cli/dry_run.py                          NEW
```

---

*Log updated: July 11, 2026, ~04:00 IST*
*Next: End-to-end live test with Fireworks API + judge demo run*
"""

with open('docs/DEVANSH_LOGS.md', 'a', encoding='utf-8') as f:
    f.write(session2_log)

print('Appended OK')
