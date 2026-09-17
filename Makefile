# Makefile — pilotage des environnements Docker Compose.
# Usage : make <cible>. Liste : make help.

COMPOSE := docker compose
LOCAL   := $(COMPOSE) --env-file .env.local #charge le fichier override .env.local pour le développement local
DEBUG   := $(COMPOSE) --env-file .env.local -f docker-compose.yml -f docker-compose.debug.yml
TEST    := $(COMPOSE) --env-file .env.test  -f docker-compose.yml -f docker-compose.test.yml
PROD    := $(COMPOSE) --env-file .env.prod  -f docker-compose.yml -f docker-compose.prod.yml

.DEFAULT_GOAL := help
.PHONY: help setup pull-models local debug test eval lint format prod down stop logs ps clean

help: ## Affiche cette aide
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Crée .env.local et .env.prod depuis .env.example (si absents)
	@test -f .env.local || (cp .env.example .env.local && echo "Créé .env.local — renseigne GROQ_API_KEY")
	@test -f .env.prod  || (cp .env.example .env.prod  && echo "Créé .env.prod")

pull-models: ## Télécharge le modèle d'embeddings Ollama (sur l'hôte)
	ollama pull bge-m3

local: ## Démarre l'environnement LOCAL (reload) — http://localhost:8000
	$(LOCAL) up --build

debug: ## Démarre DEBUG : debugpy :5678 (attend le client) + Portainer :9000
	$(DEBUG) up --build

test: ## Lance la suite Pytest dans un environnement jetable
	$(TEST) run --rm backend pytest

eval: ## Évalue le retrieval RAG (recall@k, MRR) → eval/report.json — nécessite des docs ingérés
	$(LOCAL) run --rm backend python -m eval.evaluate

lint: ## Ruff lint (dans le conteneur backend)
	$(LOCAL) run --rm --no-deps backend ruff check .

format: ## Ruff format (dans le conteneur backend)
	$(LOCAL) run --rm --no-deps backend ruff format .

prod: ## Démarre l'environnement PRODUCTION (détaché, image buildée)
	$(PROD) up -d --build

down: ## Arrête et supprime conteneurs + réseau du projet
	$(COMPOSE) down

stop: ## Arrête les conteneurs sans les supprimer
	$(COMPOSE) stop

logs: ## Suit les logs du backend
	$(COMPOSE) logs -f backend

ps: ## Liste les conteneurs du projet
	$(COMPOSE) ps

clean: ## Down + suppression des volumes (⚠️ efface la base de données)
	$(COMPOSE) down -v
