from typing import Optional

from app.config import voyage_client, bm25_model
from qdrant_client import models


async def generate_embeddings(text: str):
    dense_emb = voyage_client.embed(
        texts=[text],
        model="voyage-multilingual-2"
    ).embeddings[0]

    sparse_emb = next(bm25_model.query_embed(text))

    return dense_emb, sparse_emb

def build_price_filter(min_price: Optional[float], max_price: Optional[float]) -> Optional[models.Filter]:
    price_conditions = []

    if min_price is not None:
        price_conditions.append(
            models.FieldCondition(
                key="price",
                range=models.Range(
                    gte=min_price
                )
            )
        )

    if max_price is not None:
        price_conditions.append(
            models.FieldCondition(
                key="price",
                range=models.Range(
                    lte=max_price
                )
            )
        )

    return price_conditions if price_conditions else None