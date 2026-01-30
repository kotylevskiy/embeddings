import base64
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException

from .config import settings
from .embeddings import InputItem, embed_texts, normalize_input
from .model import ModelBundle, load_model_bundle, resolve_device
from .supported_models import load_supported_model_ids
from .schemas import EmbeddingItem, EmbeddingRequest, EmbeddingResponse, Usage


app = FastAPI(title="embeddings", version=settings.service_version)


def _get_bundle(model_id: str) -> ModelBundle:
    try:
        return load_model_bundle(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _format_embedding(embedding: List[float], encoding_format: str) -> List[float] | str:
    if encoding_format == "float":
        return embedding
    if encoding_format == "base64":
        array = np.asarray(embedding, dtype=np.float32)
        return base64.b64encode(array.tobytes()).decode("ascii")
    raise HTTPException(status_code=400, detail="encoding_format must be 'float' or 'base64'")


def _apply_dimensions(embeddings: List[List[float]], dimensions: int | None) -> List[List[float]]:
    if dimensions is None:
        return embeddings
    if not embeddings:
        return embeddings
    base_dim = len(embeddings[0])
    if dimensions > base_dim:
        raise HTTPException(
            status_code=400,
            detail=f"dimensions must be <= {base_dim}",
        )
    if dimensions == base_dim:
        return embeddings
    return [vector[:dimensions] for vector in embeddings]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    device = resolve_device()
    return {
        "service": "embeddings",
        "version": settings.service_version,
        "device": str(device),
        "cuda_memory_fraction": settings.cuda_memory_fraction,
        "max_loaded_models": settings.max_loaded_models,
        "max_batch_size": settings.max_batch_size,
        "max_input_tokens": settings.max_input_tokens,
        "truncate_long_inputs": settings.truncate_long_inputs,
        "supported_models": load_supported_model_ids(),
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    if not request.model:
        raise HTTPException(status_code=400, detail="model is required")

    try:
        inputs: List[InputItem] = normalize_input(request.input)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not inputs:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if len(inputs) > settings.max_batch_size:
        raise HTTPException(status_code=400, detail="input exceeds MAX_BATCH_SIZE")

    bundle = _get_bundle(request.model)
    try:
        embeddings, token_counts = embed_texts(
            inputs,
            bundle.tokenizer,
            bundle.model,
            bundle.device,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    embeddings = _apply_dimensions(embeddings, request.dimensions)

    data = []
    for index, embedding in enumerate(embeddings):
        data.append(
            EmbeddingItem(
                index=index,
                embedding=_format_embedding(embedding, request.encoding_format),
            )
        )

    prompt_tokens = sum(token_counts)
    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )
