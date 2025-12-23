"""带注释的 LightGCN 组件，方便快速理解。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    """
    最小化的 LightGCN 实现，并附详细注释。

    属性说明
    ----------
    num_users / num_items: int
        数据集中用户、物品的数量，对应嵌入表规模。
    embedding_dim: int
        用户/物品向量的潜在维度。
    n_layers: int
        信息传递层数，层数越高聚合的邻居越多。
    adjacency: torch.sparse_coo_tensor
        预构建的对称稀疏邻接矩阵（用户<->物品）。
    """
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3, adjacency=None, device='cpu'):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device

        # 将用户和物品统一放在一个节点空间，用户在前，物品在后。
        self.N = num_users + num_items
        self.user_offset = 0
        self.item_offset = num_users

        self.embeddings = nn.Embedding(self.N, embedding_dim)
        nn.init.normal_(self.embeddings.weight, std=0.01)

        # 邻接矩阵应由调用方预先计算并搬到目标设备。
        self.register_buffer("adjacency_indices", None)
        self.register_buffer("adjacency_values", None)
        self.register_buffer("adjacency_size", torch.tensor([0,0]))
        if adjacency is not None:
            self.set_adjacency(adjacency)

    def set_adjacency(self, adjacency):
        """接收 CPU/CUDA 上的稀疏邻接矩阵并拆分保存。"""
        if not adjacency.is_sparse:
            raise ValueError("adjacency must be a sparse_coo_tensor")
        # 拆成索引和值便于保存/加载。
        self.adjacency_indices = adjacency._indices().to(self.device)
        self.adjacency_values = adjacency._values().to(self.device)
        self.adjacency_size = torch.tensor(adjacency.shape, device=self.device)

    def _sparse_mm(self, x):
        """稀疏矩阵乘法：adjacency @ x，x 形状为 N x dim。"""
        indices = self.adjacency_indices
        values = self.adjacency_values
        size = tuple(self.adjacency_size.tolist())
        adj = torch.sparse_coo_tensor(indices, values, size, device=self.device)
        return torch.sparse.mm(adj, x)

    def forward(self):
        """
        在图上传播嵌入，最后拆出用户/物品视图。

        Returns
        -------
        user_emb: torch.Tensor
            最终用户嵌入，形状 (num_users, dim)。
        item_emb: torch.Tensor
            最终物品嵌入，形状 (num_items, dim)。
        """
        x = self.embeddings.weight  # (N, dim)
        all_embeddings = [x]

        # 逐层进行稀疏聚合。
        for layer in range(self.n_layers):
            x = self._sparse_mm(x)
            all_embeddings.append(x)

        # 对所有层（含初始 0 层）的嵌入取平均。
        out = torch.stack(all_embeddings, dim=1).mean(dim=1)  # (N, dim)

        # 将统一节点嵌入重新切分成用户/物品子集。
        user_emb = out[self.user_offset : self.user_offset + self.num_users]
        item_emb = out[self.item_offset : self.item_offset + self.num_items]
        return user_emb, item_emb

class BPRLoss:
    """
    带正则项说明的 BPR 损失封装。
    """
    def __init__(self, reg=1e-4):
        self.reg = reg

    def __call__(self, user_emb, pos_emb, neg_emb):
        # 成对排序目标：让正样本得分高于负样本。
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # 对所有嵌入做 L2 正则，避免梯度爆炸。
        reg_loss = self.reg * (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / user_emb.size(0)
        return loss + reg_loss