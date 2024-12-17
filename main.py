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

@app.get("/debug-cors")
async def debug_cors():
    logger.info("CORS test endpoint called")
    return {"message": "CORS is working"}

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


@app.get("/organizations/{org_id}/products/search")
async def search_products(
        query: str = Query(..., description="Search query text"),
        limit: int = Query(50, ge=25, le=50, description="Number of results to return")
):
    try:
        dense_query_vector, sparse_query_vector = await generate_embeddings(query)

        first_stage_results = qdrant_client.query_points(
            collection_name="gorgia_products",
            with_payload=True,
            limit=200,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(**sparse_query_vector.as_object()),
                    using="sparse",
                    limit=100,
                    score_threshold=0.42
                ),
                models.Prefetch(
                    query=dense_query_vector,
                    using="dense",
                    limit=100,
                    score_threshold=0.42
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.DBSF
            )
        )
        logger.info(f"Query: {query}")
        logger.info(f"First stage results count: {len(first_stage_results.points)}")
        if first_stage_results.points:
            logger.info(f"First result sample:")
        results = []
        for point in first_stage_results.points:
            features = point.payload.get("features", {})
            result = {
                "id": point.id,
                "name": point.payload.get("name"),
                "price": point.payload.get("price"),
                "code": point.payload.get("code"),
                "product_url": point.payload.get("product_url"),
                "image_url": point.payload.get("image_url"),
                "base_score": point.score,
                "features": features
            }

            final_score = point.score

            if features.get("ბრენდი") and \
                    features["ბრენდი"] in query:
                final_score *= 1.25

            if features.get("დასახელება") and \
                    features["დასახელება"] in query:
                final_score *= 1.35

            dimensions = ["ფერი", "ქვეყანა", "წონა", "მოცულობა", "სიგრძე", "სიგანე"]
            for dim in dimensions:
                if features.get(dim) and dim in query:
                    final_score *= 1.25
                    break

            result["final_score"] = final_score
            results.append(result)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        if results:
            logger.info(f"Top result: {results[0]['name']} (score: {results[0]['final_score']})")

        return {
            "total": len(results),
            "results": results[:limit],
            "query_analysis": {
                "processed_query": query,
                "total_candidates": len(first_stage_results.points),
                "returned_results": min(limit, len(results))
            }
        }

    except Exception as e:
        logger.error(f"Error in search_products: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/debug/collection-info")
async def get_collection_info():
    try:
        collection_info = qdrant_client.get_collection("gorgia_products")
        points_count = qdrant_client.count("gorgia_products")
        return {
            "collection_info": collection_info,
            "points_count": points_count
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)