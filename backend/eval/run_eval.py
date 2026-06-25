#!/usr/bin/env python3
"""Run RAG evaluation against golden_set.json using ragas metrics."""

import asyncio
import json
import sys
from pathlib import Path

# Allow running from backend/ or repo root
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

from services.embedder import load_model  # noqa: E402
from services.pipeline import run_pipeline  # noqa: E402


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


def run_ragas_eval(rows: list[dict]) -> dict:
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


def print_summary(rows: list[dict], scores) -> None:
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
    print(
        f"{'AVERAGE':<40} "
        f"{df['faithfulness'].mean():>8.3f} "
        f"{df['answer_relevancy'].mean():>8.3f} "
        f"{df['context_precision'].mean():>8.3f}"
    )
    domain_rate = sum(1 for r in rows if r["domain_hit"]) / len(rows) * 100
    print(f"Domain hit rate: {domain_rate:.1f}%")
    print("=" * 72)


async def main() -> None:
    if not settings.tavily_api_key or not settings.openrouter_api_key:
        print("Error: set TAVILY_API_KEY and OPENROUTER_API_KEY in backend/.env")
        sys.exit(1)

    load_model()

    golden_path = Path(__file__).parent / "golden_set.json"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # default 3 for quick runs
    print(f"Loading golden set ({limit} queries)...")

    rows = await collect_results(golden_path, limit=limit)
    if not rows:
        print("No successful pipeline runs to evaluate.")
        sys.exit(1)

    print("\nScoring with ragas...")
    scores = run_ragas_eval(rows)
    print_summary(rows, scores)


if __name__ == "__main__":
    asyncio.run(main())
