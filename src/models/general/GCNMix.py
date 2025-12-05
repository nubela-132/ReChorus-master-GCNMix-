# 基于 LightGCN 的稀疏邻接传播，并加入 IMix（正负样本混合）策略

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from models.BaseModel import GeneralModel


class GCNMixBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Embedding size.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of propagation layers.')
        parser.add_argument('--self_loop', type=int, default=0,
                            help='Whether to add self loop when building adj matrix.')
        parser.add_argument('--mix_alpha', type=float, default=0.5,
                            help='Positive/negative embedding mix ratio.')
        parser.add_argument('--mix_prob', type=float, default=0.3,
                            help='Probability to apply IMix for a training instance.')
        return parser

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        """从训练集交互构建对称归一化的稀疏邻接矩阵"""
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1.0
        R = R.tolil()

        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1)) + 1e-10
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

        if selfloop_flag:
            norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        else:
            norm_adj_mat = normalized_adj_single(adj_mat)
        return norm_adj_mat.tocsr()

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.mix_alpha = args.mix_alpha
        self.mix_prob = args.mix_prob
        self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set,
                                          selfloop_flag=bool(args.self_loop))
        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        self.encoder = GCNMixEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers,
                                     device=self.device)

    def _mix_items(self, item_embeddings, mix_mask):
        """
        item_embeddings: [B, C, D], C>=1, 第一个位置是正样本，后面是负样本
        mix_mask: [B] 的bool张量，标记哪些样本应用 IMix
        """
        if item_embeddings.size(1) <= 1:
            return item_embeddings
        batch_size = item_embeddings.size(0)
        cand_num = item_embeddings.size(1)
        rand_neg_idx = torch.randint(1, cand_num, (batch_size,), device=item_embeddings.device)
        pos_emb = item_embeddings[:, 0, :]
        neg_pick = item_embeddings[torch.arange(batch_size, device=item_embeddings.device), rand_neg_idx, :]
        mixed_pos = self.mix_alpha * pos_emb + (1 - self.mix_alpha) * neg_pick
        mixed_pos = torch.where(mix_mask[:, None], mixed_pos, pos_emb)
        return torch.cat([mixed_pos.unsqueeze(1), item_embeddings[:, 1:, :]], dim=1)

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']
        u_embed, i_embed = self.encoder(user, items)

        if self.training and self.mix_prob > 0 and feed_dict.get('phase', '') == 'train':
            mix_mask = torch.rand(u_embed.size(0), device=i_embed.device) < self.mix_prob
            i_embed = self._mix_items(i_embed, mix_mask)

        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, C]
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}


class GCNMix(GeneralModel, GCNMixBase):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'mix_alpha', 'mix_prob', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser = GCNMixBase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return GCNMixBase.forward(self, feed_dict)


class GCNMixEncoder(nn.Module):
    def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3, device='cpu'):
        super(GCNMixEncoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.device = device

        self.embedding_dict = self._init_model()
        self.register_buffer('sparse_norm_adj', self._convert_sp_mat_to_sp_tensor(norm_adj).to(self.device))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
        })
        return embedding_dict

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, users, items):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], dim=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        user_embeddings = user_all_embeddings[users.long(), :]
        item_embeddings = item_all_embeddings[items.long(), :]
        return user_embeddings, item_embeddings
