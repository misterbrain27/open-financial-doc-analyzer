"""RAG retrieval evaluation: recall@k and MRR on `eval/questions.jsonl`.

Replays each question from the evaluation set against HYBRID search (`retrieval.hybrid_search`,
dense + lexical RRF fusion, Phase 9) and measures *retrieval* quality BEFORE generation:

- recall@k: is the right page somewhere in the top-k? (binary per question)
- MRR     : at what rank does the first right page come out? (1/rank, averaged)

Ground truth is at the **page** level: a result is relevant if its `page_number` appears in the
question's `expected_pages`. No company/year filter is applied on purpose, to measure raw
retrieval (see the Phase 8 decision).

**Vector-only baseline** (Phase 8, before the hybrid): recall@5=0.933, MRR=0.889 — kept in
`doc/phase-8-eval.md` and this file's git history, to compare before/after (Phase 9).

Output: a console report (one line per question + summary) and a reproducible `eval/report.json`
(overall metrics + per-question detail) to version/compare runs.

Note: the console report and `report.json` verdict text are intentionally kept in FRENCH — this
is the actual artifact/report content (matching the app's French-facing product), not a comment.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.db import async_session
from app.retrieval.search import hybrid_search

K = 5
QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
REPORT_PATH = Path(__file__).parent / "report.json"


def load_questions() -> list[dict]:
    """Loads the JSONL evaluation set (one complete JSON object per line)."""
    questions: list[dict] = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # tolerate any blank lines
                questions.append(json.loads(line))
    return questions


def build_verdict(recall: float, mrr: float, num_miss: int, num_weak: int) -> str:
    """Readable qualitative conclusion derived from the metrics (thresholds tuned for the demo
    corpus). Returned text is in French — it's the actual report content, not a comment."""
    if recall >= 0.95 and mrr >= 0.90:
        quality = "excellent"
    elif recall >= 0.80 and mrr >= 0.70:
        quality = "bon"
    elif recall >= 0.60:
        quality = "moyen"
    else:
        quality = "faible"

    verdict = (
        f"Retrieval {quality} : la bonne page est dans le top-{K} pour {recall:.0%} des questions "
        f"(recall@{K}={recall:.3f}) et remonte en tête la plupart du temps (MRR={mrr:.3f})."
    )
    if num_miss or num_weak:
        verdict += (
            f" À corriger : {num_miss} ratée(s) et {num_weak} mal classée(s) "
            f"(piste : reranking cross-encoder, cf. stretch Phase 9)."
        )
    else:
        verdict += " Aucun cas problématique."
    return verdict


async def evaluate() -> None:
    questions = load_questions()
    records: list[dict] = []

    async with async_session() as session:
        for q in questions:
            results = await hybrid_search(q["question"], session, k=K)  # no filter: raw
            retrieved_pages = [
                r.chunk.page_number for r in results
            ]  # ordered by rank (1 = top)
            expected = set(q["expected_pages"])

            # recall@k: at least one expected page present in the top-k results.
            recall = 1.0 if expected.intersection(retrieved_pages) else 0.0

            # reciprocal rank: 1/rank of the FIRST relevant page (1-indexed rank), 0 if absent.
            reciprocal_rank = 0.0
            for rank, page in enumerate(retrieved_pages, start=1):
                if page in expected:
                    reciprocal_rank = 1.0 / rank
                    break

            records.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "expected_pages": sorted(expected),
                    "retrieved_pages": retrieved_pages,
                    "recall": recall,
                    "reciprocal_rank": round(reciprocal_rank, 3),
                }
            )

            status = "OK  " if recall else "MISS"
            print(
                f"[{status}] {q['id']:>3} | recall@{K}={recall:.0f} RR={reciprocal_rank:.2f} "
                f"| attendu={sorted(expected)} retrouvé={retrieved_pages}"
            )

    n = len(records)
    mean_recall = sum(r["recall"] for r in records) / n if n else 0.0
    mrr = sum(r["reciprocal_rank"] for r in records) / n if n else 0.0

    num_miss = sum(1 for r in records if r["recall"] == 0)
    num_weak = sum(1 for r in records if 0 < r["reciprocal_rank"] < 1)
    verdict = build_verdict(mean_recall, mrr, num_miss, num_weak)

    print("-" * 70)
    print(f"Questions   : {n}")
    print(f"recall@{K}    : {mean_recall:.3f}")
    print(f"MRR         : {mrr:.3f}")
    print(f"\nConclusion  : {verdict}")

    report = {
        "k": K,
        "num_questions": n,
        "recall_at_k": round(mean_recall, 3),
        "mrr": round(mrr, 3),
        "verdict": verdict,
        "results": records,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nRapport écrit dans {REPORT_PATH.name}")


if __name__ == "__main__":
    asyncio.run(evaluate())
