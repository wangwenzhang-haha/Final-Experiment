# src/model/soft_prompt_projector.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SoftPromptProjectorConfig:
    input_dim: int
    hidden_dim: int
    num_tokens: int
    intermediate_dim: int


class SoftPromptProjector(nn.Module):
    """
    Project a structural embedding into a soft prompt embedding.

    Args:
        input_dim: Dimension of the structure embedding.
        hidden_dim: Hidden size of the target LLM embeddings.
        num_tokens: Number of soft prompt tokens to generate.

    Input:
        tensor of shape (input_dim,) or (batch, input_dim).

    Output:
        tensor of shape (num_tokens, hidden_dim) for single inputs,
        or (batch, num_tokens, hidden_dim) for batched inputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_tokens: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_tokens <= 0:
            raise ValueError("input_dim, hidden_dim, and num_tokens must be positive.")

        intermediate_dim = max(input_dim, hidden_dim)
        self.config = SoftPromptProjectorConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            intermediate_dim=intermediate_dim,
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, num_tokens * hidden_dim),
        )

    def forward(self, structure_embedding: torch.Tensor) -> torch.Tensor:
        if structure_embedding.dim() == 1:
            structure_embedding = structure_embedding.unsqueeze(0)
        if structure_embedding.dim() != 2:
            raise ValueError("structure_embedding must be 1D or 2D tensor.")

        projected = self.mlp(structure_embedding)
        batch_size = projected.size(0)
        projected = projected.view(batch_size, self.config.num_tokens, self.config.hidden_dim)
        if batch_size == 1:
            return projected.squeeze(0)
        return projected

    def save(self, path: str) -> None:
        payload: Dict[str, Any] = {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device | None = None) -> "SoftPromptProjector":
        payload = torch.load(path, map_location=map_location)
        config = SoftPromptProjectorConfig(**payload["config"])
        model = cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_tokens=config.num_tokens,
        )
        model.load_state_dict(payload["state_dict"])
        return model


__all__ = ["SoftPromptProjector", "SoftPromptProjectorConfig"]
