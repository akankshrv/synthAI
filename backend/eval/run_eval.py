#!/usr/bin/env python3
"""Run RAG evaluation against golden_set.json using ragas metrics."""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from core.config import settings  # noqa: E402
from datasets import Dataset  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    faithfulness,
)

from services.chroma_store import init_chroma, purge_expired_chunks  # noqa: E402
from services.embedder import load_model  # noqa: E402
from services.pipeline import run_pipeline  # noqa: E402
from services.url_cache import init_redis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthAI RAG evaluation")
    parser.add_argument("limit", nargs="?", type=int, default=3, help="Number of queries")
    parser.add_argument("--save-baseline", action="store_true", help="Save results JSON")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-mmr", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--no-decompose", action="store_true")
    return parser.parse_args()


def apply_ablation_flags(args: argparse.Namespace) -> dict[str, bool]:
    flags = {
        "enable_rerank": not args.no_rerank,
        "enable_bm25": not args.no_bm25,
        "enable_cache": not args.no_cache,
        "enable_mmr": not args.no_mmr,
        "enable_dedup": not args.no_dedup,
        "enable_query_decomposition": not args.no_decompose,
    }
    for key, value in flags.items():
        setattr(settings, key, value)
    return flags


def _domain_hit(urls: list[str], expected_domains: list[str]) -> bool:
    for url in urls:
        for domain in expected_domains:
            if domain in url:
                return True
    return False


async def collect_results(golden_path: Path, limit: int | None = None) -> list[dict]:
    cases = json.loads(golden_path.read_text())
    if limit:
        cases = cases[:limit]

    rows: list[dict] = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] Running: {case['query']}")
        result = await run_pipeline(case["query"])
        if result["error"]:
            print(f"  Skipped — {result['error']}")
            continue

        rows.append(
            {
                "question": case["query"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": case["expected_summary"],
                "domain_hit": _domain_hit(result["urls"], case["expected_domains"]),
            }
        )
    return rows


def run_ragas_eval(rows: list[dict]):
    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in rows
        ]
    )

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.openrouter_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
        )
    )

    for metric in (faithfulness, answer_relevancy, context_precision):
        metric.llm = judge

    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )


def print_summary(rows: list[dict], scores) -> dict:
    df = scores.to_pandas()

    print("\n" + "=" * 72)
    print(f"{'Query':<40} {'Faith':>8} {'AnsRel':>8} {'CtxPrec':>8} {'Domain':>8}")
    print("-" * 72)

    for i, row in enumerate(rows):
        faith = df.iloc[i].get("faithfulness", float("nan"))
        ans_rel = df.iloc[i].get("answer_relevancy", float("nan"))
        ctx_prec = df.iloc[i].get("context_precision", float("nan"))
        domain = "yes" if row["domain_hit"] else "no"
        q = row["question"][:38] + ".." if len(row["question"]) > 40 else row["question"]
        print(f"{q:<40} {faith:>8.3f} {ans_rel:>8.3f} {ctx_prec:>8.3f} {domain:>8}")

    print("-" * 72)
    averages = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
    }
    print(
        f"{'AVERAGE':<40} "
        f"{averages['faithfulness']:>8.3f} "
        f"{averages['answer_relevancy']:>8.3f} "
        f"{averages['context_precision']:>8.3f}"
    )
    domain_rate = sum(1 for r in rows if r["domain_hit"]) / len(rows) * 100
    print(f"Domain hit rate: {domain_rate:.1f}%")
    print("=" * 72)

    return {**averages, "domain_hit_rate": domain_rate}


def save_baseline(flags: dict, averages: dict, rows: list[dict]) -> Path:
    out_dir = BACKEND_ROOT / settings.eval_baseline_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"baseline_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "flags": flags,
        "averages": averages,
        "query_count": len(rows),
        "embedding_model": settings.embedding_model,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nBaseline saved to {path}")
    return path


async def main() -> None:
    args = parse_args()
    flags = apply_ablation_flags(args)

    if not settings.tavily_api_key or not settings.openrouter_api_key:
        print("Error: set TAVILY_API_KEY and OPENROUTER_API_KEY in backend/.env")
        sys.exit(1)

    load_model()
    init_chroma()
    purge_expired_chunks()
    try:
        init_redis()
    except Exception as exc:
        if settings.enable_cache:
            print(f"Warning: Redis unavailable ({exc}); cache disabled for eval")
            settings.enable_cache = False

    print(f"Eval flags: {flags}")
    golden_path = Path(__file__).parent / "golden_set.json"
    print(f"Loading golden set ({args.limit} queries)...")

    rows = await collect_results(golden_path, limit=args.limit)
    if not rows:
        print("No successful pipeline runs to evaluate.")
        sys.exit(1)

    print("\nScoring with ragas...")
    scores = run_ragas_eval(rows)
    averages = print_summary(rows, scores)

    if args.save_baseline:
        save_baseline(flags, averages, rows)


if __name__ == "__main__":
    asyncio.run(main())
