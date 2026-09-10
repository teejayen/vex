# Choosing the classification models

Shortlist, prices and the defaults wired into `screenshots classify`. The model list and
pricing were pulled from `GET https://openrouter.ai/api/v1/models` on **7 September 2026**.
That endpoint needs no API key, so the numbers here can be refreshed at any time.

All prices are US dollars per million tokens.

## What the tiers need

| Tier | Needs | Typical call |
| --- | --- | --- |
| `rules` | nothing | free, no network |
| `text` | JSON output, decent reading comprehension | about 1,100 input tokens, 180 output |
| `vision` | image input **and** JSON output | about 1,100 input tokens including the 512 px thumbnail, 180 output |

The system prompt is roughly 820 tokens on its own, so the taxonomy dominates the input
regardless of how much OCR text a screenshot carries. That matters: it flattens the price
difference between a screenshot with 50 characters of text and one with 2,500.

## Shortlist: vision-capable, JSON-capable, cheapest first

Filtered to models that accept image input, advertise `response_format`, and are not
`:batch` variants (the batch endpoints are asynchronous and no use to a synchronous CLI).

| Model | Input | Output | Strict JSON schema | Notes |
| --- | --- | --- | --- | --- |
| `nex-agi/nex-n2-mini` | 0.025 | 0.100 | yes | cheapest on the list, little track record |
| `qwen/qwen3.7-flash` | 0.030 | 0.130 | no (json_object only) | 1M context |
| `google/gemma-3-4b-it` | 0.050 | 0.100 | yes | very small, weak on dense UI screenshots |
| `google/gemma-3-12b-it` | 0.050 | 0.150 | yes | good value, open weights |
| `qwen/qwen3.5-flash-02-23` | 0.065 | 0.260 | yes | |
| `mistralai/mistral-small-3.2-24b-instruct` | 0.075 | 0.200 | yes | strong on documents and OCR-style input |
| `bytedance-seed/seed-1.6-flash` | 0.075 | 0.300 | yes | |
| `google/gemma-4-26b-a4b-it` | 0.070 | 0.340 | yes | |
| `openai/gpt-5-nano` | 0.050 | 0.400 | yes | reasoning is mandatory, so output tokens balloon |
| `google/gemini-2.5-flash-lite` | 0.100 | 0.400 | yes | plus $0.0000001 per image |
| `openai/gpt-4.1-nano` | 0.100 | 0.400 | yes | |
| `qwen/qwen3-vl-8b-instruct` | 0.117 | 0.455 | yes | purpose-built vision-language model |
| `google/gemini-3.1-flash-lite` | 0.250 | 1.500 | yes | the sensible step up in quality |
| `openai/gpt-4o-mini` | 0.150 | 0.600 | yes | older, well understood |

## Shortlist: text tier, cheapest capable

Every model above can also serve the text tier. These are the cheapest text-only options
that advertise JSON output, for the case where you want to split the tiers apart.

| Model | Input | Output | Notes |
| --- | --- | --- | --- |
| `mistralai/mistral-nemo` | 0.019 | 0.030 | cheapest listed with JSON support |
| `inclusionai/ling-3.0-flash` | 0.021 | 0.063 | no strict schema |
| `openai/gpt-oss-20b` | 0.030 | 0.130 | reasoning model, output tokens run long |
| `mistralai/mistral-small-24b-instruct-2501` | 0.050 | 0.080 | 32k context |
| `meta-llama/llama-3.1-8b-instruct` | 0.050 | 0.080 | |
| `deepseek/deepseek-v4-flash-0731` | 0.050 | 0.100 | 1.3M context |
| `qwen/qwen3-30b-a3b-instruct-2507` | 0.048 | 0.193 | |

## The defaults, and why

```
text tier:   google/gemini-2.5-flash-lite
vision tier: google/gemini-2.5-flash-lite
```

The reasoning is that **cost is not the binding constraint here**. Classifying the whole
15,000-screenshot library costs somewhere between one and eight dollars across the entire
shortlist. Chasing the cheapest model saves a couple of dollars once, and costs real time
every time a reply comes back as prose instead of JSON or picks `other` for a bank
statement. So the choice is made on reliability, and price is only used to rule out
anything gratuitously expensive.

Gemini 2.5 Flash Lite wins on four counts:

- **One model, both tiers.** It is multimodal, so the text and vision tiers behave the
  same way and there is one set of quirks to learn rather than two.
- **Strict JSON schema support.** It advertises `structured_outputs`, so the client sends
  the full response schema rather than a bare `json_object` hint. Fewer malformed replies
  means fewer wasted calls.
- **Built for high throughput.** It is the cheap, fast tier of a mainstream family, which
  is exactly the shape of this job: 15,000 small independent calls at concurrency 6.
- **Dense screenshot reading.** Screenshots are small text on busy backgrounds. The 4B and
  12B open-weight models are cheaper but noticeably weaker at this, and the vision tier is
  precisely the case where OCR already failed to produce usable text.

Models deliberately not chosen as the default:

- `openai/gpt-5-nano` and the other reasoning-mandatory models. Reasoning tokens are billed
  as output, and output is where these models are expensive. A classification task with a
  fixed schema does not need a chain of thought.
- Anything `:batch`. Those endpoints are asynchronous.
- `nex-agi/nex-n2-mini` and the other very cheap unknowns. Fine as bakeoff candidates, but
  not something to point at the whole library unmeasured.

**Upgrade path.** `google/gemini-3.1-flash-lite` is the nominal step up on price and
capability, at about three times the cost. The first bakeoff did not support that, so see
the results below before reaching for it.

**Downgrade path.** `google/gemma-3-12b-it` halves the cost again, and led the first
bakeoff on both tiers. It is the leading candidate to replace the default once there is
enough evidence to move.

## What it costs to run

Assuming the auto tiering splits the library roughly 20 percent rules, 60 percent text
only, 15 percent text then escalated to vision, and 5 percent straight to vision:

| Model used for both tiers | 1,700 screenshots | 15,000 screenshots |
| --- | --- | --- |
| `google/gemini-2.5-flash-lite` (default) | $0.31 | $2.69 |
| `google/gemini-3.1-flash-lite` | $0.91 | $8.01 |
| `mistralai/mistral-small-3.2-24b-instruct` | $0.20 | $1.76 |
| `google/gemma-3-12b-it` | $0.14 | $1.22 |
| `qwen/qwen3-vl-8b-instruct` | $0.35 | $3.12 |

Worst case, with every single screenshot going through the vision tier and the rules tier
catching nothing, the default costs about $2.79 for 15,000 images. The tiering saves time
and rate limit headroom far more than it saves money.

Two things make the real bill lower than these numbers. The rules tier is free, and
`classify_cache` means a re-run against the same model never pays twice. `--dry-run` prints
the tier split and a live-priced estimate before you commit to anything.

## First bakeoff, 7 September 2026

Twenty screenshots, five candidates, both tiers. Measured cost is per image.
`<root>/bakeoff/bakeoff-20260907T214109.json` has the full run.

| Model | Tier | OK | Agreement | Median latency | Cost per image |
| --- | --- | --- | --- | --- | --- |
| `google/gemma-3-12b-it` | text | 20/20 | 85% | 1.60s | $0.000068 |
| `google/gemma-3-12b-it` | vision | 20/20 | 85% | 1.58s | $0.000076 |
| `qwen/qwen3-vl-8b-instruct` | text | 20/20 | 85% | 1.35s | $0.000243 |
| `qwen/qwen3-vl-8b-instruct` | vision | 20/20 | 80% | 1.57s | $0.000236 |
| `google/gemini-2.5-flash-lite` (default) | text | 20/20 | 80% | 1.07s | $0.000143 |
| `google/gemini-2.5-flash-lite` (default) | vision | 20/20 | 75% | 2.00s | $0.000271 |
| `mistralai/mistral-small-3.2-24b-instruct` | text | 20/20 | 80% | 1.55s | $0.000097 |
| `mistralai/mistral-small-3.2-24b-instruct` | vision | 20/20 | 70% | 2.00s | $0.000107 |
| `google/gemini-3.1-flash-lite` | text | 17/20 | 65% | 1.50s | $0.000471 |
| `google/gemini-3.1-flash-lite` | vision | 20/20 | 65% | 2.23s | $0.000729 |

**Gemma 3 12B leads on both tiers at the lowest cost of the five.** It is also the only
candidate that scores the same in text and vision, which is what you want when the auto
tiering sends the same library through both. That makes it the model to re-test, not yet
the model to switch to.

**The default has not changed.** One twenty-image sample scored against a majority vote is
not enough to move it. Three things need checking first:

- **Agreement is not accuracy.** A five-model majority can be wrong together. Gemma leading
  by one image out of twenty is inside the noise, and a five-point gap on a sample this
  size is about one screenshot.
- **The sample is small and unstratified.** Twenty images cannot cover a whole taxonomy.
  The categories that actually matter here, `recipe` and the `structured` payload it
  carries, may not have appeared at all.
- **`structured` output was not scored.** Agreement compares categories only. A model that
  picks the right category but writes a thin or malformed recipe body is worse in practice
  than the number suggests.

The step-up candidate did badly: Gemini 3.1 Flash Lite scored lowest on both tiers and
failed three of twenty text calls outright. Treat the upgrade path above as unproven.

Next run, on a larger and deliberately mixed sample:

```bash
vex bakeoff --sample 60 --models google/gemma-3-12b-it,google/gemini-2.5-flash-lite,qwen/qwen3-vl-8b-instruct
```

Then read fifty captions by eye, check the recipe rows specifically, and switch the default
only if Gemma still leads.

## Running the bakeoff

```bash
vex bakeoff --sample 20
vex bakeoff --sample 30 --models google/gemini-2.5-flash-lite,google/gemma-3-12b-it,mistralai/mistral-small-3.2-24b-instruct
```

It samples screenshots that have both a thumbnail and OCR text, runs every candidate in
every tier it supports, and prints agreement, median latency and measured cost per image
alongside projections for 1,700 and 15,000 images. The full run is dumped to
`<root>/bakeoff/bakeoff-<timestamp>.json`, and every response is written to
`classify_cache`, so a later `classify --model <winner>` reuses the calls the bakeoff
already paid for.

Read the agreement column carefully. It scores each model against the majority category
across all models on the same screenshot, not against a hand-labelled truth set. It tells
you which model is the outlier, not which one is right. Pair it with the caption column in
the JSON dump and check twenty of them by eye before switching the default.

## Changing the defaults

In order of precedence:

```bash
vex classify --model google/gemini-3.1-flash-lite   # both tiers, this run only
```

```bash
export VEX_TEXT_MODEL=google/gemini-2.5-flash-lite
export VEX_VISION_MODEL=google/gemini-3.1-flash-lite
```

```toml
# <root>/vex.toml
[classify]
text_model = "google/gemini-2.5-flash-lite"
vision_model = "google/gemini-3.1-flash-lite"
concurrency = 6
```
