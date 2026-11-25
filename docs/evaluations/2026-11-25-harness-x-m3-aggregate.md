# Harness X — Aggregate M3 Comparison (2026-11-25)

This is a small live evaluation over six real public X fixtures. It is a
diagnostic run, not a ground-truth quality benchmark.

## Setup

- Model: `tencent/hy4-flash`
- Reasoning: disabled (`--no-reasoning`); with reasoning enabled this model
  consumed the full output budget and returned no JSON for extraction.
- Prices: $0.035 / million input tokens, $0.1405 / million output tokens
- Budget per case: $0.05
- LLM path: `LLMTriageAgent` + `LLMExtractionAgent` + `LLMVerifierAgent` + `LLMArbiterAgent`
- Rule path: all four roles rule-based
- Fixtures: six real X posts in `fixtures/x_posts/`

Command:

```bash
python scripts/compare_harness_agents.py \
  --fixture fixtures/x_posts \
  --limit 10 \
  --model tencent/hy4-flash \
  --input-price-per-million 0.035 \
  --output-price-per-million 0.1405 \
  --max-cost-usd 0.05 \
  --out runs/m3 \
  --comparison-id m3-tencent-001 \
  --llm-extraction \
  --llm-arbiter \
  --no-reasoning
```

## Aggregate result

- Posts: **6**
- Agreement: **2/6 (33.3%)**
- Rule accepted / review / rejected / failed: **3 / 1 / 2 / 0**
- LLM accepted / review / rejected / failed: **4 / 2 / 0 / 0**
- Rule cost: **$0.000000**
- LLM cost: **$0.000666**
- Rule latency: **0.3 ms**
- LLM latency: **29,054.1 ms**

## Per-post result

| Post | Rule | LLM | Agreement |
|---|---|---|---|
| `1807221515089506334` | `NEEDS_REVIEW / ROAD` | `COMPLETED / ROAD` | No |
| `2076110894149652629` | `REJECTED / ROAD` | `NEEDS_REVIEW / ROAD` | No |
| `931149896203169792` | `COMPLETED / DRAINAGE_FLOOD` | `COMPLETED / ROAD` | No |
| `2026159892885832085` | `COMPLETED / ROAD` | `COMPLETED / ROAD` | Yes |
| `2125560260849193400` | `REJECTED / ROAD` | `NEEDS_REVIEW / DRAINAGE_FLOOD` | No |
| `2116307635217838493` | `COMPLETED / DRAINAGE_FLOOD` | `COMPLETED / DRAINAGE_FLOOD` | Yes |

## Observations

- The BaseBDG road post is the main safety signal. The rule path sends it to
  review because there is no specific street. The LLM accepts it with the vague
  location `jalan raya di daerah abi`; the verifier notes the vagueness but still
  considers the evidence supported. The LLM path is too permissive on location
  specificity in this run.
- The Stasiun Bandung post repeats the root-cause disagreement: the rule sees
  "banjir" and chooses drainage, while the LLM chooses road damage.
- The weather post is rejected by the rule path and sent to review by the LLM.
  The LLM path is more conservative there, avoiding a false rejection.
- The Dago accident post is rejected by the rule path and sent to review by the
  LLM. This is a disagreement about whether an accident report is actionable.

## Model cost note

Earlier smoke runs used `anthropic/claude-haiku-5.5` and
`deepseek/deepseek-v4.1-flash-1017`. They are not a controlled benchmark: the
Haiku run covered two posts, and the DeepSeek run left reasoning enabled and
produced empty extraction output for two posts. Tencent Hy4 Flash with reasoning
disabled completed all six posts for $0.000666.

## Limitations

- Six posts, no human ground truth, one model, one run.
- LLM latency includes serial calls; four LLM roles per post are much slower than
  the rule path.
- Live X search was unavailable; fixtures were captured earlier.
- The agreement metric is status + category equality, not semantic correctness.

Machine-readable result: `2026-11-25-harness-x-m3-aggregate.json`.
