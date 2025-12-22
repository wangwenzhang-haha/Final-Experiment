# src/model/lightgcn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    """
    Minimal LightGCN implementation.
    - num_users, num_items: counts
    - embedding_dim: latent size
    - n_layers: number of propagation layers
    - adjacency: torch.sparse_coo_tensor adjacency (shape (N,N)) where N = num_users + num_items
      adjacency should be symmetric (user<->item).
    """
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3, adjacency=None, device='cpu'):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device

        self.N = num_users + num_items
        self.user_offset = 0
        self.item_offset = num_users

        self.embeddings = nn.Embedding(self.N, embedding_dim)
        nn.init.normal_(self.embeddings.weight, std=0.01)

        # adjacency should be precomputed and moved to device by caller
        self.register_buffer("adjacency_indices", None)
        self.register_buffer("adjacency_values", None)
        self.register_buffer("adjacency_size", torch.tensor([0,0]))
        if adjacency is not None:
            self.set_adjacency(adjacency)

    def set_adjacency(self, adjacency):
        """
        Accepts a torch.sparse_coo_tensor adjacency on cpu or cuda.
        """
        if not adjacency.is_sparse:
            raise ValueError("adjacency must be a sparse_coo_tensor")
        # store as components to be safe for saving/loading
        self.adjacency_indices = adjacency._indices().to(self.device)
        self.adjacency_values = adjacency._values().to(self.device)
        self.adjacency_size = torch.tensor(adjacency.shape, device=self.device)

    def _sparse_mm(self, x):
        """
        Sparse matrix multiply: adjacency @ x
        adjacency represented by indices and values.
        x: (N, dim)
        """
        indices = self.adjacency_indices
        values = self.adjacency_values
        size = tuple(self.adjacency_size.tolist())
        adj = torch.sparse_coo_tensor(indices, values, size, device=self.device)
        return torch.sparse.mm(adj, x)

    def forward(self):
        """
        Propagate embeddings and return final user and item embeddings.
        Returns:
            user_emb_final: (num_users, dim)
            item_emb_final: (num_items, dim)
        """
        x = self.embeddings.weight  # (N, dim)
        all_embeddings = [x]

        for layer in range(self.n_layers):
            x = self._sparse_mm(x)
            all_embeddings.append(x)

        # average embeddings from all layers (including 0)
        out = torch.stack(all_embeddings, dim=1).mean(dim=1)  # (N, dim)

        user_emb = out[self.user_offset : self.user_offset + self.num_users]
        item_emb = out[self.item_offset : self.item_offset + self.num_items]
        return user_emb, item_emb

class BPRLoss:
    """
    BPR loss wrapper
    """
    def __init__(self, reg=1e-4):
        self.reg = reg

    def __call__(self, user_emb, pos_emb, neg_emb):
        # user_emb, pos_emb, neg_emb: (batch, dim)
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularization
        reg_loss = self.reg * (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / user_emb.size(0)
        return loss + reg_loss