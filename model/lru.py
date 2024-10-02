import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class LRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = LRUEmbedding(self.args)
        self.model = LRUModel(self.args)

    def forward(self, x):
        x, mask = self.embedding(x)
        embedding_weight = self.embedding.projector(
            torch.cat([self.embedding.token_lang.weight, self.embedding.token_img.weight], dim=-1))
        return self.model(x, embedding_weight, mask)


class LRUEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.token_lang = nn.Embedding.from_pretrained(self.load_item_embedding(),
                                                       freeze=self.args.freeze_embed,
                                                       padding_idx=0)
        self.token_img = nn.Embedding.from_pretrained(self.load_item_img_embedding(),
                                                      freeze=self.args.freeze_embed,
                                                      padding_idx=0)
        
        print(f"**** Notice: Embedding Freezing is {self.args.freeze_embed}! ****")
        self.projector = nn.Linear(
            self.token_lang.embedding_dim+self.token_img.embedding_dim,
            args.mrl_hidden_sizes[-1]
        )
        self.embed_dropout = nn.Dropout(args.lru_dropout)

    def get_mask(self, x):
        return (x > 0)

    def load_item_embedding(self):
        path = self.args.dataset._get_preprocessed_embeddings_path()
        print(f'loaded embedding from {path}')
        embeddings = torch.tensor(np.load(path))
        pad_embedding = torch.zeros(1, embeddings.size(1))
        return torch.cat([pad_embedding, embeddings], dim=0)
    
    def load_item_img_embedding(self):
        path = self.args.dataset._get_preprocessed_img_embeddings_path()
        print(f'loaded embedding from {path}')
        embeddings = torch.tensor(np.load(path))
        pad_embedding = torch.zeros(1, embeddings.size(1))
        return torch.cat([pad_embedding, embeddings], dim=0)

    def forward(self, x):
        mask = self.get_mask(x)
        x_lang, x_img = self.token_lang(x), self.token_img(x)
        x = self.projector(torch.cat([x_lang, x_img], dim=-1))
        return self.embed_dropout(x), mask


class LRUModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.mrl_hidden_sizes[-1]
        self.lru_blocks = nn.ModuleList([LRUBlock(self.args) for _ in range(args.lru_num_blocks)])

    def forward(self, x, embedding_weight, mask):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        x = x[:, -seq_len:]
        
        # prediction layer
        scores = torch.cat([torch.matmul(x[...,:s], embedding_weight[...,:s].permute(1,0))
                            for _, s in enumerate(self.args.mrl_hidden_sizes)], dim=0)
        return scores


class LRUBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_size = args.mrl_hidden_sizes[-1]
        self.input_layernorm = MRLLayerNorm(args)
        self.post_lru_layernorm = MRLLayerNorm(args)
        self.lru_layer = LRULayer(
            args, d_model=hidden_size, 
            mrl_hidden_sizes=args.mrl_hidden_sizes,
            use_bias=args.lru_use_bias, dropout=args.lru_attn_dropout, 
            r_min=args.lru_r_min, r_max=args.lru_r_max
        )
        self.feed_forward = PositionwiseFeedForward(
            args,
            mrl_hidden_sizes=args.mrl_hidden_sizes,
            use_bias=args.lru_use_bias, dropout=args.lru_dropout
        )
    
    def forward(self, x, mask):
        residual = x
        hidden = self.input_layernorm(x)
        hidden = self.lru_layer(hidden, mask)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_lru_layernorm(hidden)
        hidden = self.feed_forward(hidden)
        return residual + hidden


class LRULayer(nn.Module):
    def __init__(self,
                 args,
                 d_model,
                 mrl_hidden_sizes,
                 use_bias=False,
                 dropout=0.1,
                 r_min=0.01,
                 r_max=0.1):
        super().__init__()
        self.args = args
        self.hidden_size = d_model
        self.mrl_hidden_sizes = mrl_hidden_sizes
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D (D is omitted)
        self.in_proj = MRLSquareLinearLayer(self.mrl_hidden_sizes, use_bias=use_bias).to(torch.cfloat)
        self.out_proj = MRLSquareLinearLayer(self.mrl_hidden_sizes, use_bias=use_bias).to(torch.cfloat)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)
        mask_ = mask.reshape(B * L // l, l)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma
        
        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        return self.dropout(self.out_proj(h).real)


class MRLLayerNorm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.start_indices = args.mrl_hidden_sizes
        self.start_indices = [0] + self.start_indices[:-1]
        self.mrl_hidden_sizes = args.mrl_hidden_sizes
        self.layernorm_sizes = [self.mrl_hidden_sizes[0]]
        self.layernorm_sizes += [self.mrl_hidden_sizes[i] - self.mrl_hidden_sizes[i-1] 
                                  for i in range(1, len(self.mrl_hidden_sizes))]
        for i, hidden_size in enumerate(self.layernorm_sizes):
            setattr(self, f"layernorm_{i}", nn.LayerNorm(hidden_size))
    
    def forward(self, hidden_states):
        nesting_logits = []
        for i, (start_idx, hidden_size) in enumerate(zip(self.start_indices, self.layernorm_sizes)):
            nesting_logits += [getattr(self, f"layernorm_{i}")(hidden_states[..., start_idx:start_idx+hidden_size])]
        return torch.cat(nesting_logits, dim=-1)


class MRLSquareLinearLayer(nn.Module):
    def __init__(self, mrl_hidden_sizes, use_bias=False):
        super().__init__()
        self.input_sizes = mrl_hidden_sizes
        self.output_sizes = [mrl_hidden_sizes[0]]
        self.output_sizes += [mrl_hidden_sizes[i] - mrl_hidden_sizes[i-1] 
                              for i in range(1, len(mrl_hidden_sizes))]
        for i, (in_size, out_size) in enumerate(zip(self.input_sizes, self.output_sizes)):
            setattr(self, f"mrl_linear_{i}", nn.Linear(in_size, out_size, bias=use_bias))

    def forward(self, hidden_states):
        nesting_logits = []
        for i, in_size in enumerate(self.input_sizes):
            nesting_logits += [getattr(self, f"mrl_linear_{i}")(hidden_states[..., :in_size])]
        return torch.cat(nesting_logits, dim=-1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, args, mrl_hidden_sizes, use_bias=False, dropout=0.1):
        super().__init__()
        self.args = args
        self.gate = MRLSquareLinearLayer(mrl_hidden_sizes, use_bias=use_bias)
        self.w_1 = MRLSquareLinearLayer(mrl_hidden_sizes, use_bias=use_bias)
        self.w_2 = MRLSquareLinearLayer(mrl_hidden_sizes, use_bias=use_bias)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate_ = self.gate(x)
        x_ = self.dropout(self.w_1(x))
        return self.dropout(self.w_2(self.activation(gate_) * x_))