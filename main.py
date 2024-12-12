from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from typing import Optional, Dict
import uuid
from datetime import datetime
import json
import os
import uvicorn
from qdrant_client.grpc import FieldCondition, Direction, Filter

from app.config import ORGANIZATIONS_FILE, voyage_client, bm25_model, logger, qdrant_client
from qdrant_client import models


def load_organizations() -> Dict:
    if os.path.exists(ORGANIZATIONS_FILE):
        with open(ORGANIZATIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_organizations(organizations: Dict):
    with open(ORGANIZATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(organizations, f, ensure_ascii=False, indent=2, default=str)


async def generate_embeddings(text: str):
    """Generate both dense and sparse embeddings"""
    dense_emb = voyage_client.embed(texts=[text], model="voyage-multilingual-2").embeddings[0]
    sparse_emb = next(bm25_model.passage_embed([text]))
    return dense_emb, sparse_emb


@asynccontextmanager
async def lifespan(app_: FastAPI):
    global organizations
    organizations = load_organizations()
    logger.info(f"Loaded {len(organizations)} organizations from file")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/organizations")
async def create_organization(org: dict):
    org_id = f"org_{str(uuid.uuid4())[:8]}"
    org_data = {
        "id": org_id,
        "name": org.get("name"),
        "description": org.get("description"),
        "created_at": datetime.now()
    }
    organizations[org_id] = org_data
    save_organizations(organizations)
    return org_data


@app.get("/organizations/{org_id}")
async def get_organization(org_id: str):
    if org_id not in organizations:
        raise HTTPException(status_code=404, detail="Organization not found")
    return organizations[org_id]


@app.get("/gorgia/products/search")
async def search_products(
        query: str = Query(..., description="Search query text"),
        limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
        category: Optional[str] = Query(None, description="Filter by category"),
        min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
        max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
        min_discount: Optional[float] = Query(None, ge=0, le=100, description="Minimum discount percentage"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        search_strategy: str = Query("rrf", description="Search strategy: dense, sparse, rrf, reranking")
):
    try:
        dense_query_vector, sparse_query_vector = await generate_embeddings(query)
        filter_conditions = []
        if category:
            filter_conditions.append(
                models.FieldCondition(
                    key="category",
                    match={"value": category}
                )
            )
        if min_price is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="price",
                    range=models.Range(
                        gte=min_price
                    )
                )
            )
        if max_price is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="price",
                    range=models.Range(
                        lte=max_price
                    )
                )
            )
        if min_discount is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="discount_percentage",
                    range=models.Range(
                        gte=min_discount
                    )
                )
            )

        search_params = {
            "collection_name": "gorgia_products",
            "limit": limit,
            "offset": offset,
            "with_payload": True
        }

        if filter_conditions:
            search_params["filter"] = models.Filter(
                must=filter_conditions
            )

        if search_strategy == "dense":
            search_params["query"] = dense_query_vector
            search_params["using"] = "dense"
            search_result = qdrant_client.query_points(**search_params)

        elif search_strategy == "sparse":
            search_params["query"] = models.SparseVector(**sparse_query_vector.as_object())
            search_params["using"] = "sparse"
            search_result = qdrant_client.query_points(**search_params)

        elif search_strategy == "rrf":
            prefetch = [
                models.Prefetch(
                    query=dense_query_vector,
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_query_vector.as_object()),
                    using="sparse",
                    limit=20,
                ),
            ]
            search_result = qdrant_client.query_points(
                **search_params,
                prefetch=prefetch,
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                )
            )

        else:
            prefetch = [
                models.Prefetch(
                    query=dense_query_vector,
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_query_vector.as_object()),
                    using="sparse",
                    limit=20,
                ),
            ]
            search_result = qdrant_client.query_points(
                **search_params,
                prefetch=prefetch,
                query=dense_query_vector,
                using="dense"
            )

        results = []
        for point in search_result.points:
            result = {
                "id": point.id,
                "score": point.score,
                **point.payload
            }
            results.append(result)

        return {
            "strategy": search_strategy,
            "total": len(results),
            "offset": offset,
            "limit": limit,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in search_products: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)