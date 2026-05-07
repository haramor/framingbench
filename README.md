# FramingBench (Anonymous Supplementary)

Materials supporting the paper currently under double-blind review.

## Contents

- `framingbench/` — the 1,213 benchmark documents (10 domains × 6 sub-collections of name-version × target). Browse directly.
- `framingbench.zip` — same documents as a zip archive, plus the two scripts below for offline use.
- `croissant.json` — Croissant 1.0 metadata describing the dataset structure (machine-readable schema).
- `run_llm.py` — minimal script to evaluate an LLM on a single (domain, target, name-version) folder.
- `compute_metrics.py` — extract rankings from raw LLM responses (via a Llama 3.1 70B judge over an OpenAI-compatible endpoint) and compute TV, Δrank, top-1, NRG against the per-(domain, target) baseline document.

## Directory layout

Each domain lives at `framingbench/<domain>/<example>/`:

```
documents/                          Real names, primary target (23 files)
documents_secondary/                Real names, secondary target (20 files)
documents_generic/                  Generic numbered labels, primary target (21 files)
documents_secondary_generic/        Generic numbered labels, secondary target (18 files)
documents_synthetic/                Fictional plausible names, primary target (21 files)
documents_secondary_synthetic/      Fictional plausible names, secondary target (18 files)
```

Each filename is `<TAG>_<technique_name>.md` for one of the 19 single techniques (P1–P3, R1–R3, Q1–Q5, M1–M8) or `combo_M1_Q5.md` / `combo_M3_Q5.md` for the two stacked combinations. The Real-primary directory additionally contains `baseline.md` (the neutral reference document `d_∅`) and `editorial.md` (the source-style framing document).

## Quick start

```
# 1. Run an LLM on every document in one folder
python run_llm.py \
    --provider openai \
    --model <model-id> \
    --documents-dir framingbench/automotive/small_suvs/documents \
    --query "I'm looking for a small SUV. What do you recommend?" \
    --output responses.json

# 2. Extract rankings via Llama 3.1 70B (any OpenAI-compatible endpoint)
#    and compute TV / Δrank / top-1 / NRG for every variant against baseline
python compute_metrics.py \
    --responses responses.json \
    --baseline-doc framingbench/automotive/small_suvs/documents/baseline.md \
    --target "2025 Mitsubishi Eclipse Cross" \
    --extractor-base-url http://localhost:8000/v1 \
    --output metrics.json
```

`run_llm.py` supports `--provider {openai, anthropic, vllm}`. Per-domain queries and target product names appear in the paper's appendix and in `croissant.json`.

## Reference

Full benchmark construction protocol, technique definitions, audit procedure, and per-domain query / target specifications are documented in the paper's appendix.
