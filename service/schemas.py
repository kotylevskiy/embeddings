from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field, StrictInt, StrictStr


EmbeddingInput = Union[
    StrictStr,
    List[StrictStr],
    List[StrictInt],
    List[List[StrictInt]],
]


class EmbeddingRequest(BaseModel):
    input: EmbeddingInput = Field(..., description="Input text(s) or token IDs")
    model: str = Field(..., description="Model ID", min_length=1)
    encoding_format: Literal["float", "base64"] = Field(
        "float",
        description="Return format for embeddings",
    )
    dimensions: Optional[int] = Field(
        None,
        description="Optional embedding size override (truncation)",
        gt=0,
    )
    user: Optional[str] = Field(
        None,
        description="End-user identifier (passed through)",
    )


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Union[List[float], str]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: Usage
