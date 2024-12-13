from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"

class Organization(BaseModel):
    name: str
    description: Optional[str] = None

class OrganizationCreate(Organization):
    pass

class OrganizationResponse(Organization):
    id: str
    created_at: datetime

class Product(BaseModel):
    name: str = Field()
    description: str = Field()
    price: float = Field()
    category: str = Field()
    organization_id: str = Field()

class ProductOut(BaseModel):
    id: str = Field()
    name: str = Field()
    description: str = Field()
    price: float = Field()
    category: str = Field()
    score: float = Field()

class SearchResult(BaseModel):
    products: List[Product]
    total: int
    page: int
    limit: int