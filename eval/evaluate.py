"""Évaluation du retrieval RAG : recall@k et MRR sur `eval/questions.jsonl`.

Rejoue chaque question du jeu d'évaluation dans la recherche HYBRIDE (`retrieval.hybrid_search`,
fusion RRF dense + lexical, Phase 9) et mesure la qualité du *retrieval* AVANT la génération :

- recall@k : la bonne page est-elle quelque part dans le top-k ? (binaire par question)
- MRR      : à quel rang sort la première bonne page ? (1/rang, moyenné)

La vérité-terrain est au niveau **page** : un résultat est pertinent si son `page_number`
figure dans les `expected_pages` de la question. On n'applique volontairement **aucun filtre**
company/year ici, afin de mesurer le retrieval brut (cf. décision Phase 8).

Baseline **vectoriel pur** (Phase 8, avant l'hybride) : recall@5=0.933, MRR=0.889 — conservée dans
`doc/phase-8-eval.md` et l'historique git de ce fichier, pour comparer avant/après (Phase 9).

Sortie : un rapport console (une ligne par question + résumé) et un `eval/report.json`
reproductible (métriques globales + détail par question) pour versionner/comparer les runs.
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
    """Charge le jeu d'évaluation JSONL (un objet JSON complet par ligne)."""
    questions: list[dict] = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # tolère d'éventuelles lignes vides
                questions.append(json.loads(line))
    return questions


def build_verdict(recall: float, mrr: float, num_miss: int, num_weak: int) -> str:
    """Conclusion qualitative lisible, dérivée des métriques (seuils adaptés au corpus de démo)."""
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
            results = await hybrid_search(
                q["question"], session, k=K
            )  # pas de filtre : brut
            retrieved_pages = [
                r.chunk.page_number for r in results
            ]  # ordonné par rang (1 = top)
            expected = set(q["expected_pages"])

            # recall@k : au moins une page attendue présente dans les k premiers résultats.
            recall = 1.0 if expected.intersection(retrieved_pages) else 0.0

            # reciprocal rank : 1/rang de la PREMIÈRE page pertinente (rang 1-indexé), 0 si absente.
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
