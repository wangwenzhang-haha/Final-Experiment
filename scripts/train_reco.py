#!/usr/bin/env python3
# scripts/train_reco.py
import os
import yaml
import argparse
import random
import pickle
import time
from pathlib import Path

import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader

from src.model.lightgcn import LightGCN, BPRLoss

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class InteractionDataset(Dataset):
    """
    Dataset for BPR-style training.
    Expects interactions as list of (user_idx, item_idx).
    """
    def __init__(self, interactions, num_items, user_pos_dict=None, neg_sample_size=1):
        self.interactions = interactions
        self.num_items = num_items
        self.neg_sample_size = neg_sample_size
        self.user_pos = user_pos_dict if user_pos_dict is not None else self._build_user_pos()

    def _build_user_pos(self):
        d = {}
        for u,i in self.interactions:
            d.setdefault(u, set()).add(i)
        return d

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        u, pos = self.interactions[idx]
        # negative sampling: sample until not in user's positives
        neg = random.randrange(0, self.num_items)
        tries = 0
        while neg in self.user_pos.get(u, set()) and tries < 10:
            neg = random.randrange(0, self.num_items)
            tries += 1
        return u, pos, neg

def build_mappings_and_interactions(G):
    """
    From fused Graph (networkx MultiDiGraph) build:
    - user2idx, item2idx
    - interactions list [(u_idx, i_idx), ...]
    """
    user_nodes = [n for n,d in G.nodes(data=True) if d.get('node_type') == 'user' or str(n).startswith("user:")]
    item_nodes = [n for n,d in G.nodes(data=True) if d.get('node_type') == 'item' or str(n).startswith("item:")]

    # fallback if not labeled node_type
    if not user_nodes:
        user_nodes = [n for n in G.nodes() if str(n).startswith("user:")]
    if not item_nodes:
        item_nodes = [n for n in G.nodes() if str(n).startswith("item:")]

    user2idx = {u: idx for idx, u in enumerate(sorted(user_nodes))}
    item2idx = {i: idx for idx, i in enumerate(sorted(item_nodes))}

    interactions = []
    for u, v, data in G.edges(data=True):
        # consider edges where u is user and v is item or vice versa
        if u in user2idx and v in item2idx:
            interactions.append((user2idx[u], item2idx[v]))
        elif v in user2idx and u in item2idx:
            interactions.append((user2idx[v], item2idx[u]))
    return user2idx, item2idx, interactions

def build_adjacency(num_users, num_items, interactions):
    """
    Build symmetric adjacency (sparse) for LightGCN.
    Returns torch.sparse_coo_tensor on CPU (caller may move to device).
    """
    import scipy.sparse as sp
    N = num_users + num_items
    rows = []
    cols = []
    data = []
    for u,i in interactions:
        ui = u
        ii = num_users + i
        rows.append(ui); cols.append(ii); data.append(1.0)
        rows.append(ii); cols.append(ui); data.append(1.0)
    mat = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    # row-normalize (symmetric normalization recommended in literature)
    row_sum = np.array(mat.sum(axis=1)).flatten()
    row_sum[row_sum == 0] = 1.0
    inv_deg = 1.0 / row_sum
    diag = sp.diags(inv_deg)
    norm_mat = diag.dot(mat)  # simple left-normalize (sufficient for toy)
    coo = norm_mat.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    size = coo.shape
    adj = torch.sparse_coo_tensor(indices, values, size)
    return adj

def collate_fn(batch):
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    pos = torch.tensor([b[1] for b in batch], dtype=torch.long)
    neg = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return users, pos, neg

def train(args):
    config = load_config(args.config)
    device = torch.device(config.get('run', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    seed = config.get('run', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load fused graph
    fused_path = args.fused_graph if args.fused_graph else config.get('run', {}).get('fused_graph', 'data/processed/fused_graph.pkl')
    if not os.path.exists(fused_path):
        raise FileNotFoundError(f"Fused graph not found: {fused_path}. Run preprocess first.")
    with open(fused_path, 'rb') as f:
        G = pickle.load(f)

    user2idx, item2idx, interactions = build_mappings_and_interactions(G)
    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"Users: {num_users}, Items: {num_items}, Interactions: {len(interactions)}")

    adj = build_adjacency(num_users, num_items, interactions)
    batch_size = args.batch_size if args.batch_size else config.get('model', {}).get('train_batch_size', 512)
    epochs = args.epochs if args.epochs else config.get('model', {}).get('epochs', 20)
    embedding_dim = config.get('model', {}).get('embedding_dim', 64)
    n_layers = config.get('model', {}).get('n_layers', 3)
    lr = config.get('model', {}).get('lr', 0.001)

    dataset = InteractionDataset(interactions, num_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = LightGCN(num_users, num_items, embedding_dim=embedding_dim, n_layers=n_layers, adjacency=adj, device=str(device))
    model.to(device)
    # ensure adjacency stored on device
    model.set_adjacency(adj.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bpr = BPRLoss(reg=1e-4)

    out_dir = Path(args.save_dir or config.get('run', {}).get('save_dir', 'outputs/checkpoints/reco'))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lightgcn.pth"

    print("Start training...")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, (users, pos, neg) in enumerate(loader):
            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            optimizer.zero_grad()
            user_emb, item_emb = model.forward()
            u_emb = user_emb[users]               # (B, dim)
            pos_emb = item_emb[pos]
            neg_emb = item_emb[neg]

            loss = bpr(u_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        t1 = time.time()
        print(f"Epoch {epoch}/{epochs} loss={epoch_loss:.4f} time={t1 - t0:.1f}s")
        # simple checkpoint per epoch
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'config': config,
            'user2idx': user2idx,
            'item2idx': item2idx
        }, str(ckpt_path))

    print(f"Training finished. Last checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml", help="config yaml")
    parser.add_argument("--fused_graph", type=str, default=None, help="path to fused_graph.pkl")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    train(args)