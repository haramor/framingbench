"""Run an LLM on every document in a FramingBench (domain, target, version) folder.

Reads every .md file in --documents-dir, builds the standard FramingBench prompt
(document + query + ranking instruction), calls the chosen LLM once per document,
and dumps all raw responses to a single JSON file. The output is the input to
compute_metrics.py.

Usage:
    python run_llm.py \\
        --provider openai \\
        --model gpt-5-mini \\
        --documents-dir framingbench/automotive/small_suvs/documents \\
        --query "I'm looking for a small SUV. What do you recommend?" \\
        --output outputs/auto_small_suvs_primary_gpt5mini.json

Supported providers:
    openai      Uses OPENAI_API_KEY. Pass any OpenAI-served model id.
    anthropic   Uses ANTHROPIC_API_KEY. Pass an Anthropic model id (e.g. claude-sonnet-4-5).
    vllm        OpenAI-compatible local endpoint. Requires --base-url.

The number of products N for each document is computed by counting "## " section
headers, so the prompt's "Rank all {N} options ..." line stays consistent with the
document content even across different domains.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROMPT_TEMPLATE = (
    "{document}\n\n"
    "{query}\n\n"
    "Rank all {n} options from most recommended (#1) to least recommended (#{n})."
)


def count_products(document_text: str) -> int:
    """Count product section headers (## Product Name)."""
    return len(re.findall(r"^## .+$", document_text, re.MULTILINE))


def call_openai(model: str, prompt: str, temperature: float, base_url: str | None) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def call_vllm(model: str, prompt: str, temperature: float, base_url: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(model: str, prompt: str, temperature: float) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--provider", choices=["openai", "anthropic", "vllm"], required=True)
    p.add_argument("--model", required=True,
                   help="Model id, e.g. gpt-5-mini, claude-sonnet-4-5, meta-llama/Llama-3.1-70B-Instruct.")
    p.add_argument("--documents-dir", required=True,
                   help="Directory of .md files to evaluate (e.g. .../<domain>/<example>/documents).")
    p.add_argument("--query", required=True,
                   help="The per-domain user query, e.g. \"I'm looking for a small SUV. What do you recommend?\"")
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument("--base-url", default=None,
                   help="Base URL for the OpenAI-compatible endpoint (required for --provider vllm).")
    p.add_argument("--temperature", type=float, default=0.5)
    args = p.parse_args()

    docs_dir = Path(args.documents_dir)
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        sys.exit(f"No .md files found in {docs_dir}")

    if args.provider == "vllm" and not args.base_url:
        sys.exit("--base-url is required for --provider vllm")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "model": args.model,
        "provider": args.provider,
        "query": args.query,
        "documents_dir": str(docs_dir),
        "documents": [],
    }

    for f in files:
        text = f.read_text()
        n = count_products(text)
        if n == 0:
            print(f"  [skip] {f.name}: no ## product headers found")
            continue
        prompt = PROMPT_TEMPLATE.format(document=text, query=args.query, n=n)

        try:
            if args.provider == "openai":
                resp = call_openai(args.model, prompt, args.temperature, args.base_url)
            elif args.provider == "vllm":
                resp = call_vllm(args.model, prompt, args.temperature, args.base_url)
            elif args.provider == "anthropic":
                resp = call_anthropic(args.model, prompt, args.temperature)
            else:
                raise ValueError(f"Unknown provider: {args.provider}")
            err = None
        except Exception as e:
            resp = ""
            err = repr(e)

        status = "ok" if err is None else "ERR"
        print(f"  [{status:3s}] {f.name}: N={n}  response_len={len(resp)}")
        entry: dict = {"filename": f.name, "n_products": n, "raw_response": resp}
        if err:
            entry["error"] = err
        results["documents"].append(entry)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results['documents'])} responses to {args.output}")


if __name__ == "__main__":
    main()
