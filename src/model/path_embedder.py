"""Path embedding utilities for structured multi-hop paths."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import torch
from torch import nn

PathTriple = Tuple[str, str, str]
Path = Sequence[PathTriple]


class PathEmbedder(nn.Module):
    """Encode structured knowledge graph paths into dense vectors."""

    def __init__(
        self,
        entity_vocab: Dict[str, int],
        relation_vocab: Dict[str, int],
        embed_dim: int = 128,
        method: str = "mean",
    ) -> None:
        super().__init__()
        if method not in {"mean", "gru"}:
            raise ValueError(f"Unsupported method '{method}'. Use 'mean' or 'gru'.")
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.embed_dim = embed_dim
        self.method = method

        self.entity_embedding = nn.Embedding(len(entity_vocab), embed_dim)
        self.relation_embedding = nn.Embedding(len(relation_vocab), embed_dim)
        nn.init.normal_(self.entity_embedding.weight, std=0.02)
        nn.init.normal_(self.relation_embedding.weight, std=0.02)

        self.entity_unk_id = self.entity_vocab.get("<unk>")
        self.relation_unk_id = self.relation_vocab.get("<unk>")

        if method == "gru":
            self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        else:
            self.gru = None

    def encode(self, paths: Union[Path, Sequence[Path]]) -> torch.Tensor:
        """Encode one or more paths into dense vectors.

        Args:
            paths: Single path or list of paths.

        Returns:
            Tensor with shape [D] for a single path or [N, D] for a batch.
        """
        if not paths:
            return torch.zeros(self.embed_dim)

        if self._is_single_path(paths):
            return self._encode_single(paths)  # type: ignore[arg-type]

        return self._encode_batch(paths)  # type: ignore[arg-type]

    def _encode_single(self, path: Path) -> torch.Tensor:
        if self.method == "mean":
            embeddings = self._collect_embeddings(path)
            if embeddings.numel() == 0:
                return torch.zeros(self.embed_dim, device=embeddings.device)
            return embeddings.mean(dim=0)

        sequence = self._path_to_sequence(path)
        if not sequence:
            return torch.zeros(self.embed_dim)
        inputs = torch.stack(sequence, dim=0).unsqueeze(0)
        _, hidden = self.gru(inputs)
        return hidden[-1, 0]

    def _encode_batch(self, paths: Sequence[Path]) -> torch.Tensor:
        if self.method == "mean":
            vectors = [self._encode_single(path) for path in paths]
            return torch.stack(vectors, dim=0)

        sequences = [self._path_to_sequence(path) for path in paths]
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        if lengths.max().item() == 0:
            return torch.zeros(len(paths), self.embed_dim)

        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def _collect_embeddings(self, path: Path) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        for head, relation, tail in path:
            embeddings.append(self.entity_embedding(self._entity_index(head)))
            embeddings.append(self.relation_embedding(self._relation_index(relation)))
            embeddings.append(self.entity_embedding(self._entity_index(tail)))
        if not embeddings:
            return torch.empty(0, self.embed_dim)
        return torch.stack(embeddings, dim=0)

    def _path_to_sequence(self, path: Path) -> List[torch.Tensor]:
        sequence: List[torch.Tensor] = []
        for idx, (head, relation, tail) in enumerate(path):
            if idx == 0:
                sequence.append(self.entity_embedding(self._entity_index(head)))
            sequence.append(self.relation_embedding(self._relation_index(relation)))
            sequence.append(self.entity_embedding(self._entity_index(tail)))
        return sequence

    def _entity_index(self, entity: str) -> torch.Tensor:
        idx = self.entity_vocab.get(entity, self.entity_unk_id)
        if idx is None:
            raise KeyError(f"Entity '{entity}' missing from entity vocabulary.")
        return torch.tensor(idx, dtype=torch.long)

    def _relation_index(self, relation: str) -> torch.Tensor:
        idx = self.relation_vocab.get(relation, self.relation_unk_id)
        if idx is None:
            raise KeyError(f"Relation '{relation}' missing from relation vocabulary.")
        return torch.tensor(idx, dtype=torch.long)

    @staticmethod
    def _is_single_path(paths: Union[Path, Sequence[Path]]) -> bool:
        if not isinstance(paths, Iterable):
            return False
        return len(paths) > 0 and isinstance(paths[0], tuple)
