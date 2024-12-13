from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from typing import Dict
import uuid
from datetime import datetime
import json
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
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

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/organizations/{org_id}/products")
async def create_product(
        org_id: str,
        product: dict
):
    if org_id not in organizations:
        raise HTTPException(status_code=404, detail="Organization not found")
    dense_emb, sparse_emb = await generate_embeddings(product.get("description", ""))
    point_id = str(uuid.uuid4())
    product_payload = {
        **product,
        "organization_id": org_id,
        "created_at": datetime.now()
    }

    qdrant_client.upsert(
        collection_name="gorgia_products",
        points=[
            models.PointStruct(
                id=point_id,
                vector=dense_emb,
                sparse_vector=models.SparseVector(**sparse_emb.as_object()),
                payload=product_payload
            )
        ]
    )

    return {
        "id": point_id,
        **product_payload
    }


@app.get("/organizations/{org_id}/products/search")
async def search_products(
        org_id: str,
        query: str = Query(..., description="Search query text"),
        limit: int = Query(50, ge=25, le=50, description="Number of results to return")
):
    if org_id not in organizations:
        raise HTTPException(status_code=404, detail="Organization not found")
    try:
        dense_query_vector, sparse_query_vector = await generate_embeddings(query)
        search_result = qdrant_client.query_points(
            collection_name="gorgia_products",
            with_payload=True,
            limit=200,
            score_threshold=0.49,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(**sparse_query_vector.as_object()),
                    using="sparse",
                    limit=150,
                    score_threshold=0.47
                ),
                models.Prefetch(
                    query=dense_query_vector,
                    using="dense",
                    limit=150,
                    score_threshold=0.47
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.DBSF
            )
        )

        results = []
        for point in search_result.points:
            if point.score >= 0.49:
                results.append({
                    "name": point.payload.get("name"),
                    "price": point.payload.get("price"),
                    "old_price": point.payload.get("old_price"),
                    "discount_percentage": point.payload.get("discount_percentage"),
                    "image_url": point.payload.get("image_url"),
                    "product_url": point.payload.get("product_url"),
                    "score": point.score
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        final_limit = max(min(limit, len(results)), 50)
        results = results[:final_limit]

        return {
            "total": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Error in search_products: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)