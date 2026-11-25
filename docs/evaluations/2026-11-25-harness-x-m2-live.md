# Harness X — Live M2 Comparison (2026-11-25)

This is a small live smoke evaluation, not a general quality benchmark.

## Setup

- Model: `anthropic/claude-haiku-5.5`
- Prices used for estimation: $0.10 / million input tokens, $0.50 / million output tokens
- Budget per case: $0.05
- Fixtures: two real public X posts in `fixtures/x_posts/`
- Rule path: `RuleTriageAgent` + `RuleExtractionAgent` + `RuleVerifierAgent` + `RuleArbiterAgent`
- LLM path: `LLMTriageAgent` + `RuleExtractionAgent` + `LLMVerifierAgent` + `RuleArbiterAgent`

Command:

```bash
python scripts/compare_harness_agents.py \
  --fixture fixtures/x_posts \
  --limit 2 \
  --model anthropic/claude-haiku-5.5 \
  --input-price-per-million 0.10 \
  --output-price-per-million 0.50 \
  --max-cost-usd 0.05 \
  --out runs/m2 \
  --comparison-id m2-live-001
```

## Results

| Post | Rule decision | LLM decision | Agreement |
|---|---|---|---|
| [Stasiun Bandung](https://x.com/i/status/931149896203169792) | `COMPLETED / DRAINAGE_FLOOD` | `COMPLETED / ROAD` | No |
| [Gatot Subroto](https://x.com/MGT_RADIO/status/2026159892885832085) | `COMPLETED / ROAD` | `COMPLETED / ROAD` | Yes |

- Agreement: **1/2 (50%)**
- Rule accepted / review / failed: **2 / 0 / 0**
- LLM accepted / review / failed: **2 / 0 / 0**
- Rule cost: **$0.000000**
- LLM cost: **$0.000769**
- Rule latency: **0.15 ms**
- LLM latency: **9,087.7 ms**

## Interpretation

The disagreement is the Stasiun Bandung post. The rule classifier sees the word
"banjir" and chooses `DRAINAGE_FLOOD`. The LLM chooses `ROAD`, treating the
flooding as an effect of the damaged road. For this post, the LLM decision looks
more consistent with the root cause, but two posts are not enough to generalize.

## Limitations

- Only two posts; the agreement rate is not a quality estimate.
- There is no human-labeled ground truth in this run.
- The live X search backend was unavailable during this period; the fixtures were
  captured earlier through Agent-Reach read/search.
- Extraction remains rule-based in this comparison.

Machine-readable result: `2026-11-25-harness-x-m2-live.json`.
