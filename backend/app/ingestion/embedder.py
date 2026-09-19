import httpx

from app.config import settings

EMBEDDING_DIM = 1024


async def embed_texts(
    texts: list[str], *, client: httpx.AsyncClient | None = None
) -> list[list[float]]:
    """Appelle l'API Ollama pour générer les embeddings des textes fournis."""
    if len(texts) == 0:
        return []
    url = settings.ollama_base_url + "/api/embed"
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()
    try:
        response = await client.post(url, json={"model": settings.embedding_model, "input": texts})
        data = response.json()  # transforme la réponse JSON en dict Python
        for vector in data["embeddings"]:
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"Modèle {settings.embedding_model!r} : dimension obtenue {len(vector)} "
                    f"!= dimension attendue {EMBEDDING_DIM}"
                )
        return data["embeddings"]
    finally:
        if own_client:
            await client.aclose()
