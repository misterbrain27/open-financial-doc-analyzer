import httpx

from app.config import settings

EMBEDDING_DIM = 1024


async def embed_texts(
    texts: list[str], *, client: httpx.AsyncClient | None = None
) -> list[list[float]]:
    """Calls the Ollama API to generate embeddings for the given texts."""
    if len(texts) == 0:
        return []
    url = settings.ollama_base_url + "/api/embed"
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()
    try:
        response = await client.post(url, json={"model": settings.embedding_model, "input": texts})
        data = response.json()  # turns the JSON response into a Python dict
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
