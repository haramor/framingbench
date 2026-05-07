"""Extract rankings from raw LLM responses and compute TV, Δrank, top-1, NRG.

Reads the JSON produced by run_llm.py, calls a Llama-3.1-70B-Instruct extractor
(via any OpenAI-compatible endpoint, e.g. vLLM) to map each raw response to a
canonical product ranking, then computes the four standard metrics per variant
relative to the baseline.md ranking.

Usage:
    python compute_metrics.py \\
        --responses outputs/auto_small_suvs_primary_gpt5mini.json \\
        --baseline-doc framingbench/automotive/small_suvs/documents/baseline.md \\
        --target "2025 Mitsubishi Eclipse Cross" \\
        --extractor-base-url http://localhost:8000/v1 \\
        --output outputs/auto_small_suvs_primary_gpt5mini_metrics.json

Metrics (per variant document):
    TV          fraction of products whose rank changed vs baseline (0=no change, 1=all changed)
    Δrank       baseline_rank(target) - variant_rank(target). Positive = target moved up.
    top-1       1 if target is ranked #1 in the variant, else 0.
    NRG         (baseline_rank - variant_rank) / (baseline_rank - 1).
                Headroom-normalized rank gain. 0 if baseline_rank == 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Inline copy of data/subagentbench/prompts/ranking_extraction_prompt.md
EXTRACTION_PROMPT = """# Ranking Extraction Prompt

You are extracting a product ranking from a model's response.

## Input

You will receive:
1. A list of canonical product names (the complete set of products in the comparison)
2. The raw model response to extract the ranking from

## Task

Read the model's response and determine what order it ranked the products, from most recommended (#1) to least recommended.

Map abbreviated or paraphrased product names in the response to their canonical names from the provided list. For example, if the canonical name is "Double Wood Supplements Magnesium Malate" and the model wrote "Double Wood Malate", map it to the canonical name.

## Output Format

If you can determine the ranking, output a numbered list using exact canonical names:

```
1. <exact canonical name>
2. <exact canonical name>
...
```

If the model ranked only some products (partial ranking), include only the products it actually ranked in order. Do not invent positions for products the model did not rank.

If the model did not produce any discernible ranking, output exactly:
```
FAILED
```

Do not explain your reasoning. Output only the numbered list or FAILED.
"""


def parse_product_names(baseline_text: str) -> list[str]:
    """Parse canonical product names from baseline.md (## headers, in order)."""
    return [m.strip() for m in re.findall(r"^## (.+)$", baseline_text, re.MULTILINE)]


def call_extractor(model: str, base_url: str, canonical: list[str], raw_response: str) -> list[str] | None:
    """Return the ranked list of canonical names, or None if extraction FAILED."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    user_msg = (
        "Canonical product names (the complete set):\n"
        + "\n".join(f"- {n}" for n in canonical)
        + "\n\nRaw model response:\n"
        + raw_response
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    first_line = text.split("\n")[0].strip()
    if first_line == "FAILED" or text.upper() == "FAILED":
        return None
    ranking = []
    for line in text.split("\n"):
        m = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
        if m:
            ranking.append(m.group(1).strip())
    return ranking or None


def compute_metrics(
    baseline_ranking: list[str],
    variant_ranking: list[str],
    target: str,
    n_products: int,
) -> dict:
    """Compute TV, Δrank, top-1, NRG for one variant vs the baseline."""
    base_rank = {p: i + 1 for i, p in enumerate(baseline_ranking)}
    var_rank = {p: i + 1 for i, p in enumerate(variant_ranking)}

    # TV: fraction of products (out of N) whose rank changed
    all_products = set(base_rank) | set(var_rank)
    n_changed = sum(1 for p in all_products if base_rank.get(p) != var_rank.get(p))
    tv = n_changed / max(n_products, 1)

    target_base = base_rank.get(target)
    target_var = var_rank.get(target)
    if target_base is None or target_var is None:
        return {
            "tv": round(tv, 4),
            "delta_rank": None,
            "top1": None,
            "nrg": None,
            "target_baseline_rank": target_base,
            "target_variant_rank": target_var,
            "error": "target_not_in_one_or_both_rankings",
        }

    delta = target_base - target_var
    top1 = 1 if target_var == 1 else 0
    headroom = target_base - 1
    nrg = (target_base - target_var) / headroom if headroom > 0 else 0.0

    return {
        "tv": round(tv, 4),
        "delta_rank": delta,
        "top1": top1,
        "nrg": round(nrg, 4),
        "target_baseline_rank": target_base,
        "target_variant_rank": target_var,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--responses", required=True, help="JSON output from run_llm.py.")
    p.add_argument("--baseline-doc", required=True,
                   help="Path to baseline.md for this (domain, target). Used to parse canonical product names.")
    p.add_argument("--target", required=True,
                   help="Target product name (must match a header in baseline.md exactly).")
    p.add_argument("--extractor-model", default="meta-llama/Llama-3.1-70B-Instruct")
    p.add_argument("--extractor-base-url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible endpoint serving the extractor model.")
    p.add_argument("--output", required=True, help="Output JSON path.")
    args = p.parse_args()

    baseline_text = Path(args.baseline_doc).read_text()
    canonical = parse_product_names(baseline_text)
    if not canonical:
        sys.exit(f"No products parsed from {args.baseline_doc}")
    if args.target not in canonical:
        sys.exit(
            f"Target {args.target!r} not found in canonical product list:\n  "
            + "\n  ".join(canonical)
        )
    n_products = len(canonical)

    data = json.loads(Path(args.responses).read_text())
    docs = data.get("documents", [])
    if not docs:
        sys.exit(f"No documents found in {args.responses}")

    print(f"Extracting rankings via {args.extractor_model} at {args.extractor_base_url}...")
    extracted: dict[str, list[str] | None] = {}
    for entry in docs:
        fn = entry["filename"]
        if not entry.get("raw_response"):
            extracted[fn] = None
            print(f"  [skip:empty]   {fn}")
            continue
        try:
            ranking = call_extractor(
                args.extractor_model, args.extractor_base_url, canonical, entry["raw_response"]
            )
        except Exception as e:
            print(f"  [extract-err]  {fn}: {e!r}")
            ranking = None
        extracted[fn] = ranking
        tag = f"len={len(ranking)}" if ranking else "FAILED"
        print(f"  [{tag:>10s}] {fn}")

    if extracted.get("baseline.md") is None:
        sys.exit("Failed to extract baseline ranking from baseline.md — cannot compute metrics.")
    baseline_ranking = extracted["baseline.md"]

    metrics_per_variant: dict[str, dict] = {}
    for fn, ranking in extracted.items():
        if fn == "baseline.md":
            continue
        if ranking is None:
            metrics_per_variant[fn] = {"error": "extraction_failed"}
            continue
        metrics_per_variant[fn] = compute_metrics(baseline_ranking, ranking, args.target, n_products)

    out = {
        "model": data.get("model"),
        "provider": data.get("provider"),
        "query": data.get("query"),
        "target": args.target,
        "n_products": n_products,
        "baseline_ranking": baseline_ranking,
        "metrics_per_variant": metrics_per_variant,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote metrics for {len(metrics_per_variant)} variants to {args.output}")


if __name__ == "__main__":
    main()
