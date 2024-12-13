import logging
import os
import voyageai
from qdrant_client import QdrantClient
from fastembed.sparse.bm25 import Bm25

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',handlers=[logging.StreamHandler(),])
logger = logging.getLogger(__name__)

QDRANT_URL="https://151d8faa-d52c-42ec-9ee6-c195e3bde77e.eu-central-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY="3edO3CK4OBriVlKxJJ4DvSrqsu-a1JPbEhmlXZp7SgmHd4T0xqHRZg"
VOYAGE_API_KEY="pa-hLlHB6pX8hZQxzMx4_RWKSOcwgARCXvVXPaA4M_p1Kg"

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
bm25_model = Bm25("Qdrant/bm25")
ORGANIZATIONS_FILE = "organizations.json"