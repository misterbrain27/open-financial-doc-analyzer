# financial-doc-analyzer

[🇬🇧 Read in English](README.md)

Système **RAG** (Retrieval-Augmented Generation) full-stack d'analyse de documents financiers
français (rapports annuels / documents réglementaires), construit comme projet vitrine — neuf
phases incrémentales, chacune livrée sur sa propre pull request revue.

**Architecture IA hybride** : embeddings locaux (Ollama `bge-m3`) + génération via **Mistral AI**,
abstraite derrière une interface de fournisseur (rebasculer sur un LLM 100 % local est un
changement d'une ligne). Stockage vectoriel **PostgreSQL + pgvector**. Backend **FastAPI**
(Python 3.12, async), frontend **Angular 21** (composants standalone, signals, `httpResource`).

## Stack technique

| Couche | Choix | Pourquoi |
|---|---|---|
| Backend | Python 3.12 + FastAPI + Uvicorn | Async, validation Pydantic, doc OpenAPI auto |
| Embeddings | Ollama `bge-m3` (multilingue, dim. 1024) | Local, gratuit, aucune donnée ne sort de la machine |
| Génération | Mistral AI (`mistral-small-latest`, SDK `mistralai`) | LLM cloud gratuit, bon support du français |
| Base vectorielle | PostgreSQL 16 + pgvector (index HNSW) | SQL + recherche vectorielle dans une seule base |
| Recherche lexicale | PostgreSQL full-text (`tsvector`, index GIN) | Fusionnée à la recherche vectorielle par RRF (Phase 9) |
| ORM | SQLAlchemy 2 (async) + `asyncpg` | ORM async moderne, types vecteurs natifs |
| Parsing PDF | `pdfplumber` | Gère les tableaux financiers, pas seulement le texte |
| Frontend | Angular 21 — standalone, signals, `httpResource` | Angular réactif moderne, sans NgModules |
| Conteneurs | Docker Compose (pattern override : local/debug/test/prod) | Reproductible sur tous les environnements |
| Qualité | Ruff (Python) · Prettier/ESLint (TypeScript) | Vérifié en CI à chaque PR |
| Tests | Pytest (backend) · Vitest + Playwright (frontend) | Unitaires + intégration + E2E |

## Architecture

```
Ingestion (une fois par document)           Requête (à chaque question)
──────────────────────────────              ────────────────────────────
PDF                                          question
 │  pdfplumber (texte + tableaux)              │  embedding (Ollama bge-m3)
 ▼                                             ▼
chunker (découpage ancré, overlap)          ┌──────────────┬───────────────┐
 │                                         recherche         recherche
 ▼                                         vectorielle       lexicale
embedder (Ollama bge-m3, dim 1024)         (pgvector,        (Postgres FTS,
 │                                         cosinus)           ts_rank_cd)
 ▼                                             └───────┬───────┘
PostgreSQL + pgvector                                  ▼
 (table chunks : embedding + tsvector)       Reciprocal Rank Fusion
                                                        │
                                                        ▼
                                            prompt groundé → Mistral AI
                                                        │
                                                        ▼
                                    réponse streamée (SSE) + sources citées
```

## Ce qui est implémenté, phase par phase

| Phase | Livre | Module clé |
|---|---|---|
| 0 — Fondations | Docker Compose multi-env (local/debug/test/prod) + Makefile, config Ruff/Pytest | `docker-compose*.yml`, `Makefile` |
| 1 — Ingestion | PDF → texte nettoyé + tableaux + métadonnées (entreprise, année) | `app/ingestion/loader.py` |
| 2 — Chunking | Découpage ancré aux limites avec overlap ; un chunk par tableau | `app/ingestion/chunker.py` |
| 3 — Embeddings & indexation | Embeddings Ollama, schéma pgvector + index HNSW | `app/ingestion/embedder.py`, `app/models.py` |
| 4 — Retrieval | Recherche vectorielle top-k, filtres metadata | `app/retrieval/search.py` |
| 5 — Génération | Interface LLM agnostique du fournisseur, prompt groundé anti-hallucination | `app/rag/llm.py`, `app/rag/pipeline.py` |
| 6 — API | `/health`, `/ingest`, `/query` (streaming SSE) | `app/api/routes.py` |
| 7 — Frontend | Interface chat + upload Angular, réponses streamées avec citations | `frontend/src/app/features/` |
| 8 — Tests & évaluation | E2E Playwright ; éval retrieval maison (recall@k, MRR) | `frontend/e2e/`, `eval/evaluate.py` |
| 9 — Recherche hybride | Recherche vectorielle + lexicale fusionnées par Reciprocal Rank Fusion | `app/retrieval/search.py` (`hybrid_search`) |

## Évaluation du retrieval — mesurer avant d'optimiser

`make eval` rejoue 15 questions à vérité-terrain connue sur la couche de retrieval et calcule
**recall@k** et **MRR** — la qualité du retrieval mesurée *avant même* que la génération n'entre en
jeu. La Phase 8 a établi une baseline avec la recherche vectorielle seule ; le résultat a motivé la
recherche hybride de la Phase 9 :

| Métrique | Vectoriel seul (Phase 8) | Hybride : vectoriel + full-text, RRF (Phase 9) |
|---|---|---|
| recall@5 | 0.933 (14/15) | **1.000 (15/15)** |
| MRR | 0.889 | **0.967** |

**Pourquoi ce gain** : les embeddings denses sont faibles sur les lookups lexicaux exacts — un
code d'identification d'entreprise ne « ressemble » à rien sémantiquement, donc les embeddings les
plus proches peuvent être totalement hors-sujet. Fusionner un signal full-text PostgreSQL léger
(aucune infrastructure nouvelle — la même image `pgvector/pgvector`) via Reciprocal Rank Fusion
rattrape ces cas sans dégrader les correspondances sémantiques qui fonctionnaient déjà.

## Tests

- **Backend** : 59 tests Pytest (unitaires + intégration sur un vrai Postgres/pgvector éphémère),
  `make test`.
- **Frontend** : tests unitaires Vitest (composants, services) + 6 scénarios E2E Playwright
  (upload → question → réponse streamée, API mockée au niveau réseau), `npm test` / `npm run e2e`.
- **CI** : GitHub Actions lance Ruff (lint + format) et la suite Pytest complète à chaque PR.

## Démarrage

```bash
make setup           # génère .env.local / .env.prod depuis .env.example
make pull-models      # télécharge le modèle d'embedding Ollama bge-m3
make local            # lance PostgreSQL + le backend FastAPI (http://localhost:8000)
make test             # suite Pytest complète, base de données jetable
make eval             # évaluation du retrieval (recall@k, MRR) — nécessite des documents ingérés
```

Frontend (à lancer séparément, non conteneurisé) :

```bash
cd frontend
npm install
npm start             # http://localhost:4200
```

Ingérer un document en CLI :

```bash
docker compose --env-file .env.local run --rm backend python scripts/ingest_cli.py data/raw/votre-fichier.pdf
```

> Les PDF de `data/raw/` sont git-ignorés.

Toutes les commandes passent par le **Makefile** — `make help` liste toutes les cibles.
