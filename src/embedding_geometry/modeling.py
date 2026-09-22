from __future__ import annotations

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, models


def load_encoder(
    model_id_or_path: str | Path,
    *,
    max_seq_length: int,
    device: str | torch.device,
    raw_backbone: bool = False,
) -> SentenceTransformer:
    source = str(model_id_or_path)
    if raw_backbone:
        transformer = models.Transformer(source, max_seq_length=max_seq_length)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(modules=[transformer, pooling, models.Normalize()], device=str(device))
    else:
        model = SentenceTransformer(source, device=str(device))
        model.max_seq_length = max_seq_length
    return model


def encode_features(model: SentenceTransformer, texts: list[str], device: torch.device) -> torch.Tensor:
    features = model.tokenize(texts)
    # SentenceTransformers 5.x may attach non-tensor metadata (for example
    # prompt-related text fields) to the feature mapping.
    features = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in features.items()
    }
    return model(features)["sentence_embedding"]


def verify_normalized(model: SentenceTransformer, texts: list[str]) -> dict[str, float]:
    values = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    norms = torch.from_numpy(values).norm(dim=1)
    return {"minimum": float(norms.min()), "maximum": float(norms.max()), "mean": float(norms.mean())}
