import json
from pathlib import Path

CAL = Path('calibration/calibration_prompts.jsonl')
prompts = [json.loads(l) for l in CAL.read_text(encoding='utf-8').splitlines() if l.strip()]
gsm = [p for p in prompts if p['source'] == 'gsm8k']

print(f'GSM8K prompts: {len(gsm)}')
for p in gsm[:5]:
    words_in_prompt = len(p['prompt'].split())
    ref = p['reference']
    diff = p.get('difficulty', '?')
    print(f'  [{diff}] words:{words_in_prompt:3d} | ref:{ref!r:6} | {p["prompt"][:65]}...')

print()
math_hard = [p for p in prompts if p['source'] == 'math_hard']
print(f'math_hard probes: {len(math_hard)}')
for p in math_hard:
    print(f'  [{p["difficulty"]}] {p["prompt"][:65]}')

l1_math = [p for p in prompts if p.get('difficulty') == 'L1' and p['source'] in ('gsm8k','math_hard')]
print(f'\nL1 math probes (simple algebra/arith that 3B models can do): {len(l1_math)}')

l2_gsm = [p for p in prompts if p['source'] == 'gsm8k' and p.get('difficulty') == 'L2']
l3_gsm = [p for p in prompts if p['source'] == 'gsm8k' and p.get('difficulty') == 'L3']
print(f'GSM8K L2 probes: {len(l2_gsm)}  (these are multi-step word problems)')
print(f'GSM8K L3 probes: {len(l3_gsm)}  (these are harder multi-step word problems)')
print()
print('Max tokens in calibration generate() call: 128')
print('Tokens needed for GSM8K step-by-step: ~150-350')
print('RESULT: model cut off before reaching final answer -> UNDERESTIMATES capability')
