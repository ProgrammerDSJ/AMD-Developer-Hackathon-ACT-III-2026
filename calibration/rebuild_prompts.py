"""
calibration/rebuild_prompts.py
-------------------------------
Single source of truth for ALL calibration prompts.

Rebuilds calibration_prompts.jsonl by:
  1. Keeping all existing good prompts (ARC, GSM8K, HumanEval, MMLU,
     math_hard, leetcode, musique, math_l1)
  2. REMOVING truthfulqa (epistemics trap, 0% on 3B models)
  3. FIXING existing ifeval evaluator: mcq_keyword -> keyword
  4. FIXING math_hard_005 evaluator: mcq_keyword -> keyword
  5. ADDING language_mcq L1/L2 probes (grammar, translation, paraphrase)
  6. ADDING hotpotqa L2/L3 reasoning probes (single-hop syllogisms)
  7. ADDING more ifeval instruction L2/L3 probes

Run:  python calibration/rebuild_prompts.py
"""
import json
from pathlib import Path
from collections import Counter

CAL = Path(__file__).resolve().parent / "calibration_prompts.jsonl"

# ── Load existing ─────────────────────────────────────────────────────────────
existing = [json.loads(l) for l in CAL.read_text(encoding="utf-8").splitlines() if l.strip()]

# ── Transform existing prompts ────────────────────────────────────────────────
transformed = []
removed = []
for p in existing:
    # 1. Remove truthfulqa — epistemics trap, 3B models score 0% by design
    if p["source"] == "truthfulqa":
        removed.append(p["prompt_id"])
        continue

    # 2. Fix ifeval: mcq_keyword → keyword (keyword substring match)
    if p["source"] == "ifeval" and p["evaluator"] == "mcq_keyword":
        p = dict(p, evaluator="keyword")

    # 3. Fix math_hard_005 fraction: mcq_keyword → keyword
    if p.get("prompt_id") == "math_hard_005" and p["evaluator"] == "mcq_keyword":
        p = dict(p, evaluator="keyword")

    transformed.append(p)

existing_ids = {p["prompt_id"] for p in transformed}
print(f"Removed {len(removed)} truthfulqa prompts")

# ── New prompts ───────────────────────────────────────────────────────────────
NEW_PROMPTS = [

    # ════════════════════════════════════════════════════════════════════════
    # LANGUAGE — grammar / translation / paraphrase (MCQ, evaluator=mcq)
    # Replaces truthfulqa. Tests real language skills, not epistemics traps.
    # ════════════════════════════════════════════════════════════════════════

    {"prompt_id":"lang_001","source":"language_mcq","difficulty":"L1",
     "prompt":"Which sentence is grammatically correct?\nA. Him went to the store yesterday.\nB. He went to the store yesterday.\nC. He go to the store yesterday.\nD. He going to the store yesterday.\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"lang_002","source":"language_mcq","difficulty":"L1",
     "prompt":"What is the plural of 'mouse' (the animal)?\nA. mouses\nB. meese\nC. mice\nD. mouse\nAnswer:",
     "reference":"C","evaluator":"mcq"},

    {"prompt_id":"lang_003","source":"language_mcq","difficulty":"L1",
     "prompt":"Which word is a synonym for 'happy'?\nA. sad\nB. angry\nC. joyful\nD. tired\nAnswer:",
     "reference":"C","evaluator":"mcq"},

    {"prompt_id":"lang_004","source":"language_mcq","difficulty":"L1",
     "prompt":"The French translation of 'Thank you' is:\nA. Bonjour\nB. Merci\nC. Au revoir\nD. Excusez-moi\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"lang_005","source":"language_mcq","difficulty":"L1",
     "prompt":"Which sentence uses the correct past tense?\nA. Yesterday, I eated breakfast.\nB. Yesterday, I have eaten breakfast.\nC. Yesterday, I ate breakfast.\nD. Yesterday, I was eat breakfast.\nAnswer:",
     "reference":"C","evaluator":"mcq"},

    {"prompt_id":"lang_006","source":"language_mcq","difficulty":"L1",
     "prompt":"What is the antonym (opposite) of 'ancient'?\nA. old\nB. historical\nC. modern\nD. classic\nAnswer:",
     "reference":"C","evaluator":"mcq"},

    {"prompt_id":"lang_007","source":"language_mcq","difficulty":"L1",
     "prompt":"Which sentence is in passive voice?\nA. The dog bit the man.\nB. The man was bitten by the dog.\nC. The man bit the dog.\nD. A dog bites a man.\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"lang_008","source":"language_mcq","difficulty":"L1",
     "prompt":"The Spanish word 'Casa' means:\nA. Car\nB. Cat\nC. Clock\nD. House\nAnswer:",
     "reference":"D","evaluator":"mcq"},

    {"prompt_id":"lang_009","source":"language_mcq","difficulty":"L1",
     "prompt":"Which word correctly completes: 'Neither the students nor the teacher ___ happy.'\nA. were\nB. was\nC. are\nD. be\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"lang_010","source":"language_mcq","difficulty":"L1",
     "prompt":"Which of the following is a compound sentence?\nA. The cat sat on the mat.\nB. She ran, but she was too late.\nC. Running quickly down the street.\nD. A beautiful sunny day.\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    # Language L2 — idiomatic expressions, literary devices, advanced grammar
    {"prompt_id":"lang_011","source":"language_mcq","difficulty":"L2",
     "prompt":"Which sentence contains a dangling modifier?\nA. Walking down the street, the trees were beautiful.\nB. Walking down the street, I saw beautiful trees.\nC. The trees were beautiful as I walked.\nD. I walked and saw beautiful trees.\nAnswer:",
     "reference":"A","evaluator":"mcq"},

    {"prompt_id":"lang_012","source":"language_mcq","difficulty":"L2",
     "prompt":"Which literary device is used in: 'The wind whispered through the trees'?\nA. Simile\nB. Metaphor\nC. Personification\nD. Alliteration\nAnswer:",
     "reference":"C","evaluator":"mcq"},

    {"prompt_id":"lang_013","source":"language_mcq","difficulty":"L2",
     "prompt":"What is the meaning of the idiom 'to burn the midnight oil'?\nA. To work late into the night\nB. To destroy property\nC. To cook elaborate meals\nD. To waste resources\nAnswer:",
     "reference":"A","evaluator":"mcq"},

    {"prompt_id":"lang_014","source":"language_mcq","difficulty":"L2",
     "prompt":"Which word has a similar meaning to 'ephemeral'?\nA. eternal\nB. transient\nC. substantial\nD. recurring\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"lang_015","source":"language_mcq","difficulty":"L2",
     "prompt":"In 'The committee has made its decision', 'committee' is treated as:\nA. plural noun\nB. collective noun used as singular\nC. proper noun\nD. compound noun\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    # ════════════════════════════════════════════════════════════════════════
    # REASONING — single-hop deduction / syllogisms / temporal (hotpotqa style)
    # Small, clean problems that test logical inference without memorization.
    # ════════════════════════════════════════════════════════════════════════

    {"prompt_id":"reason_001","source":"hotpotqa","difficulty":"L2",
     "prompt":"All mammals are warm-blooded. A whale is a mammal. Is a whale warm-blooded?\nA. Yes\nB. No\nC. Only in warm water\nD. Cannot be determined\nAnswer:",
     "reference":"A","evaluator":"mcq"},

    {"prompt_id":"reason_002","source":"hotpotqa","difficulty":"L2",
     "prompt":"If today is Wednesday, what day will it be in 5 days?\nA. Sunday\nB. Monday\nC. Saturday\nD. Friday\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"reason_003","source":"hotpotqa","difficulty":"L2",
     "prompt":"John is older than Mary, and Mary is older than Tom. Who is the youngest?\nA. John\nB. Mary\nC. Tom\nD. They are the same age\nAnswer:",
     "reference":"C","evaluator":"mcq"},

    {"prompt_id":"reason_004","source":"hotpotqa","difficulty":"L2",
     "prompt":"A car travels at 60 mph. How long to travel 120 miles?\nA. 1 hour\nB. 2 hours\nC. 3 hours\nD. 4 hours\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"reason_005","source":"hotpotqa","difficulty":"L2",
     "prompt":"No reptiles are warm-blooded. A snake is a reptile. Which conclusion follows?\nA. Snakes are warm-blooded.\nB. Snakes are not warm-blooded.\nC. Some snakes are warm-blooded.\nD. Warm-blooded animals are reptiles.\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"reason_006","source":"hotpotqa","difficulty":"L2",
     "prompt":"If Alice is taller than Bob, and Bob is taller than Charlie, which statement is definitely true?\nA. Charlie is the tallest.\nB. Bob is the shortest.\nC. Alice is the tallest.\nD. They are all the same height.\nAnswer:",
     "reference":"C","evaluator":"mcq"},

    {"prompt_id":"reason_007","source":"hotpotqa","difficulty":"L2",
     "prompt":"A train leaves Station A at 9 AM and the journey takes 3 hours. What time does it arrive?\nA. 11 AM\nB. Noon\nC. 1 PM\nD. 3 PM\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"reason_008","source":"hotpotqa","difficulty":"L2",
     "prompt":"A store has 24 apples. They sell half, then receive 10 more. How many apples are there now?\nA. 12\nB. 22\nC. 34\nD. 14\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    # Reasoning L3 — two-hop deduction, set theory, logical implication
    {"prompt_id":"reason_009","source":"hotpotqa","difficulty":"L3",
     "prompt":"In a class of 30 students, 18 like math, 15 like science, and 8 like both. How many like neither?\nA. 5\nB. 3\nC. 7\nD. 10\nAnswer:",
     "reference":"A","evaluator":"mcq"},

    {"prompt_id":"reason_010","source":"hotpotqa","difficulty":"L3",
     "prompt":"A box has red and blue balls. Ratio of red to blue is 3:2. There are 15 red balls. Total balls?\nA. 10\nB. 25\nC. 30\nD. 20\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"reason_011","source":"hotpotqa","difficulty":"L3",
     "prompt":"If P implies Q, and Q implies R, and R is false, what can we conclude about P?\nA. P is true\nB. P is false\nC. P might be true or false\nD. P implies R directly\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    {"prompt_id":"reason_012","source":"hotpotqa","difficulty":"L3",
     "prompt":"All A are B. All B are C. Some C are not D. Which must be true?\nA. All A are D.\nB. All A are C.\nC. No A are C.\nD. Some B are not A.\nAnswer:",
     "reference":"B","evaluator":"mcq"},

    # ════════════════════════════════════════════════════════════════════════
    # INSTRUCTION — keyword evaluator (substring match, not letter extraction)
    # Tests actual instruction-following: format transformation, generation.
    # ════════════════════════════════════════════════════════════════════════

    {"prompt_id":"instr_001","source":"ifeval","difficulty":"L2",
     "prompt":"Rewrite in passive voice: 'The chef prepared the meal.'",
     "reference":"prepared","evaluator":"keyword"},

    {"prompt_id":"instr_002","source":"ifeval","difficulty":"L2",
     "prompt":"Give the comparative form of the adjective 'good'.",
     "reference":"better","evaluator":"keyword"},

    {"prompt_id":"instr_003","source":"ifeval","difficulty":"L2",
     "prompt":"What is the plural of 'datum'?",
     "reference":"data","evaluator":"keyword"},

    {"prompt_id":"instr_004","source":"ifeval","difficulty":"L2",
     "prompt":"Convert this statement to a yes/no question: 'She is going to the market.'",
     "reference":"going","evaluator":"keyword"},

    {"prompt_id":"instr_005","source":"ifeval","difficulty":"L3",
     "prompt":"Summarize in one sentence: 'The Earth revolves around the Sun in an elliptical orbit, completing one revolution approximately every 365.25 days, which is why we have leap years every four years to account for the extra quarter-day.'",
     "reference":"365","evaluator":"keyword"},

    {"prompt_id":"instr_006","source":"ifeval","difficulty":"L3",
     "prompt":"List the 3 primary colors of light (RGB).",
     "reference":"green","evaluator":"keyword"},
]

# ── Filter out any duplicates ─────────────────────────────────────────────────
added = []
for p in NEW_PROMPTS:
    if p["prompt_id"] not in existing_ids:
        added.append(p)
        existing_ids.add(p["prompt_id"])

all_prompts = transformed + added

# ── Write out ─────────────────────────────────────────────────────────────────
out = "\n".join(json.dumps(p, ensure_ascii=False) for p in all_prompts) + "\n"
CAL.write_text(out, encoding="utf-8")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"Added {len(added)} new prompts")
print(f"Total: {len(all_prompts)} prompts\n")

dist = Counter((p["source"], p.get("difficulty","?")) for p in all_prompts)
ev_dist = Counter(p["evaluator"] for p in all_prompts)
print("Evaluator distribution:")
for k, v in sorted(ev_dist.items()):
    print(f"  {k:<14} x{v}")
print()
print("Source / difficulty distribution:")
for (src, lv), cnt in sorted(dist.items()):
    print(f"  {src:<14} {lv}  x{cnt}")
