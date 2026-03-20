import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )


class Attention(nn.Module):
    def __init__(
            self,
            heads=8,
            dim=64,
            dropout=0.,
            inner_dim=0,
    ):
        super().__init__()

        self.heads = heads
        if inner_dim == 0:
            inner_dim = dim
        self.scale = (inner_dim / heads) ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_out=False):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)

        if attn_out:
            return out, attn
        else:
            return out


class ProdAttention(nn.Module):
    def __init__(
            self,
            heads=8,
            dim=64,
            dropout=0.,
            inner_dim=0,
            topk=-1,
    ):
        super().__init__()

        self.heads = heads
        if inner_dim == 0:
            inner_dim = dim

        # if topk = -1:
        self.dim = dim
        self.topk = topk

        self.scale = (inner_dim / heads) ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.2)
        )

    def forward(self, x, attn_out=False):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        value, idx = torch.topk(attn, dim=-1, k=self.topk)  # bs group num_per_group
        idx = idx.unsqueeze(-1).repeat((1, 1, 1, 1, self.dim // h))
        vv = v.unsqueeze(-2).repeat((1, 1, 1, self.topk, 1))
        xx_ = torch.gather(vv, 2, idx)

        # =================== output ===================
        x = xx_.sum(dim=-2)
        x = (x - x.min()) / (x.max() - x.min())
        out = torch.exp(x)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.out(out)

        return out


class MemoryBlock(nn.Module):
    def __init__(
            self,
            token_num,
            heads,
            dim,
            attn_dropout,
            cluster,
            target_mode,
            groups,
            num_per_group,
            use_cls_token,
            sum_or_prod=None,
            qk_relu=False) -> None:
        super().__init__()

        if num_per_group == -1:
            self.num_per_group = -1
            # Do not use grouping, calculate for all
        else:
            self.num_per_group = max(math.ceil(token_num / groups), num_per_group)
            num_per_group = max(math.ceil(token_num / groups), num_per_group)
            self.gather_layer = nn.Conv1d((groups + int(use_cls_token)) * num_per_group, groups + int(use_cls_token),
                                          groups=groups + int(use_cls_token), kernel_size=1)

        self.soft = nn.Softmax(dim=-1)
        self.qk_relu = qk_relu
        self.dropout = nn.Dropout(attn_dropout)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(attn_dropout)
        )

        self.groups = groups
        self.use_cls_token = int(use_cls_token)
        # self.num_per_group = num_per_group
        self.heads = heads
        self.target_mode = target_mode
        self.cluster = cluster

        if cluster:
            if target_mode == 'mix':
                self.target_token = nn.Parameter(torch.rand([groups, dim]))
                self.to_target = nn.Linear(groups + token_num + int(use_cls_token), groups + int(use_cls_token))
            else:
                self.target_token = nn.Parameter(torch.rand([groups + int(use_cls_token), dim]))

        if sum_or_prod not in ['sum', 'prod']:
            print('{} is not in [sum, prod]'.format(sum_or_prod))
            raise ValueError
        self.sum_or_prod = sum_or_prod
        self.scale = dim / heads

    def forward(self, x):
        b, l, d = x.shape
        h = self.heads

        if self.sum_or_prod == 'prod':
            x = torch.log(nn.ReLU()(x) + 1)
        target = self.target_token
        target = target.reshape(1, -1, d).repeat((b, 1, 1))

        # =================== define qkv ===================
        if self.cluster:
            if self.target_mode == 'mix':
                target = torch.cat([target, x], dim=-2)
                target = self.to_target(target.transpose(-1, -2)).transpose(-1, -2)
            q = self.q(target)
        else:
            q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.reshape(b, -1, h, d // h).permute(0, 2, 1, 3)
        k = k.reshape(b, -1, h, d // h).permute(0, 2, 1, 3)
        v = v.reshape(b, -1, h, d // h).permute(0, 2, 1, 3)

        # if self.qk_relu:
        #     q = nn.ReLU()(q)
        #     k = nn.ReLU()(k)

        # =================== cak attn ===================
        attn = self.soft(
            torch.matmul(q, k.transpose(-1, -2)) * (self.scale ** -0.5)
        )
        attn = self.dropout(attn)

        # =================== gather relative tokens ===================
        if self.num_per_group == -1:
            x = einsum('b h i j, b h j d -> b h i d', attn, v)
            # x = rearrange(x, 'b h n d -> b n (h d)', h = h)
        else:
            # ===== find topk related for each group =====
            value, idx_original = torch.topk(attn, dim=-1, k=self.num_per_group)  # bs head group num_per_group

            # ===== apply summation =====
            idx = idx_original.unsqueeze(-1).repeat((1, 1, 1, 1, d // h))
            vv = v.unsqueeze(-2).repeat((1, 1, 1, self.num_per_group, 1))
            xx_ = torch.gather(vv, 2, idx)

            # ???????????????????????????????
            # x = xx_.sum(dim=-2)
            x = self.gather_layer(xx_.reshape(b * h, -1, d // h)).reshape(b, h, -1, d // h)

            # flag_map = torch.zeros_like(attn) #bs head group token
            # flag_map = flag_map.scatter_(-1,idx_original,1)
            # v_ = v.unsqueeze(dim=2).repeat(1,1,self.groups+1,1,1)
            # flag_map_ = flag_map.unsqueeze(-1).repeat(1,1,1,1,d//h)
            # xx_ = flag_map_ * v_
            # x = xx_.sum(dim=-2)

        # =================== output ===================

        if self.sum_or_prod == 'prod':
            x = (x - x.min()) / (x.max() - x.min())
            x = torch.exp(x)
        out = rearrange(x, 'b h n d -> b n (h d)', h=h)
        out = self.out(out)

        return out


class MemoryBlock2(nn.Module):
    def __init__(
            self,
            token_num,
            heads,
            dim,
            attn_dropout,
            cluster,
            target_mode,
            groups,
            num_per_group,
            use_cls_token,
            sum_or_prod=None,
            qk_relu=False) -> None:
        super().__init__()

        if num_per_group == -1:
            self.num_per_group = -1
            # Do not use grouping, calculate for all
        else:
            self.num_per_group = max(math.ceil(token_num / groups), num_per_group)
            num_per_group = max(math.ceil(token_num / groups), num_per_group)
            self.gather_layer = nn.Conv1d((groups + int(use_cls_token)) * num_per_group, groups + int(use_cls_token),
                                          groups=groups + int(use_cls_token), kernel_size=1)
            self.gather_layer = nn.Conv1d(num_per_group, 1, kernel_size=1)

        self.soft = nn.Softmax(dim=-1)
        self.qk_relu = qk_relu
        self.dropout = nn.Dropout(attn_dropout)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(attn_dropout)
        )

        self.groups = groups
        self.use_cls_token = int(use_cls_token)
        # self.num_per_group = num_per_group
        self.heads = heads
        self.target_mode = target_mode
        self.cluster = cluster

        if cluster:
            if target_mode == 'mix':
                self.target_token = nn.Parameter(torch.rand([groups, dim]))
                self.to_target = nn.Linear(groups + token_num + int(use_cls_token), groups + int(use_cls_token))
            else:
                self.target_token = nn.Parameter(torch.rand([groups + int(use_cls_token), dim]))

        if sum_or_prod not in ['sum', 'prod']:
            print('{} is not in [sum, prod]'.format(sum_or_prod))
            raise ValueError
        self.sum_or_prod = sum_or_prod
        self.scale = dim / heads

    def forward(self, x):
        b, l, d = x.shape
        h = self.heads

        if self.sum_or_prod == 'prod':
            x = torch.log(nn.ReLU()(x) + 1)
        target = self.target_token
        target = target.reshape(1, -1, d).repeat((b, 1, 1))

        # =================== define qkv ===================
        if self.cluster:
            if self.target_mode == 'mix':
                target = torch.cat([target, x], dim=-2)
                target = self.to_target(target.transpose(-1, -2)).transpose(-1, -2)
            q = self.q(target)
        else:
            q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.reshape(b, -1, h, d // h).permute(0, 2, 1, 3)
        k = k.reshape(b, -1, h, d // h).permute(0, 2, 1, 3)
        v = v.reshape(b, -1, h, d // h).permute(0, 2, 1, 3)

        # if self.qk_relu:
        #     q = nn.ReLU()(q)
        #     k = nn.ReLU()(k)

        # =================== cak attn ===================
        attn = self.soft(
            torch.matmul(q, k.transpose(-1, -2)) * (self.scale ** -0.5)
        )
        attn = self.dropout(attn)

        # =================== gather relative tokens ===================
        if self.num_per_group == -1:
            x = einsum('b h i j, b h j d -> b h i d', attn, v)
            # x = rearrange(x, 'b h n d -> b n (h d)', h = h)
        else:
            # ===== find topk related for each group =====
            value, idx_original = torch.topk(attn, dim=-1, k=self.num_per_group)  # bs head group num_per_group

            # ===== apply summation =====
            idx = idx_original.unsqueeze(-1).repeat((1, 1, 1, 1, d // h))
            vv = v.unsqueeze(-2).repeat((1, 1, 1, self.num_per_group, 1))
            xx_ = torch.gather(vv, 2, idx)

            # x = self.gather_layer(xx_.reshape(b*h, -1, d//h)).reshape(b, h, -1, d//h) #torch.Size([1024, 8, 55, 24])
            # x =
            x = value.unsqueeze(-1).repeat(1, 1, 1, 1, d // h) * xx_
            x = self.gather_layer(x.reshape(-1, 2, 24)).reshape(b, h, -1, d // h)

            # flag_map = torch.zeros_like(attn) #bs head group token
            # flag_map = flag_map.scatter_(-1,idx_original,1)
            # v_ = v.unsqueeze(dim=2).repeat(1,1,self.groups+1,1,1)
            # flag_map_ = flag_map.unsqueeze(-1).repeat(1,1,1,1,d//h)
            # xx_ = flag_map_ * v_
            # x = xx_.sum(dim=-2)

        # =================== output ===================

        if self.sum_or_prod == 'prod':
            x = (x - x.min()) / (x.max() - x.min())
            x = torch.exp(x)
        out = rearrange(x, 'b h n d -> b n (h d)', h=h)
        out = self.out(out)

        return out

# transformer

class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            attn_dropout,
            ff_dropout,
            use_cls_token,
            groups,
            sum_num_per_group,
            prod_num_per_group,
            cluster,
            target_mode,
            token_num,
            token_descent=False,
            use_prod=True,
            qk_relu=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        flag = int(use_cls_token)

        if not token_descent:
            groups = [token_num for _ in groups]

        for i in range(depth):
            token_num = token_num if i == 0 else groups[i - 1]
            self.layers.append(nn.ModuleList([
                MemoryBlock(
                    token_num=token_num,
                    heads=heads,
                    dim=dim,
                    attn_dropout=attn_dropout,
                    cluster=cluster,
                    target_mode=target_mode,
                    groups=groups[i],
                    num_per_group=prod_num_per_group[i],
                    use_cls_token=use_cls_token,
                    sum_or_prod='prod',
                    qk_relu=qk_relu) if use_prod else nn.Identity(),
                MemoryBlock(
                    token_num=token_num,
                    heads=heads,
                    dim=dim,
                    attn_dropout=attn_dropout,
                    cluster=cluster,
                    target_mode=target_mode,
                    groups=groups[i],
                    num_per_group=sum_num_per_group[i],
                    use_cls_token=use_cls_token,
                    sum_or_prod='sum',
                    qk_relu=qk_relu) if token_descent else Attention(heads=heads, dim=dim, dropout=attn_dropout),
                nn.Linear(2 * (groups[i] + flag), groups[i] + flag),
                nn.Linear(token_num + flag, groups[i] + flag) if token_descent else nn.Identity(),
                FeedForward(dim, dropout=ff_dropout),
            ]))
        self.use_prod = use_prod

    def forward(self, x):

        for toprod, tosum, down, downx, ff in self.layers:

            attn_out = tosum(x)
            if self.use_prod:
                prod = toprod(x)
                attn_out = down(torch.cat([attn_out, prod], dim=1).transpose(2, 1)).transpose(2, 1)

            x = attn_out + downx(x.transpose(-1, -2)).transpose(-1, -2)
            x = ff(x) + x

        return x


# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases


# main class

class FTTransformer(nn.Module):
    def __init__(
            self,
            args,
    ):
        super().__init__()
        '''
        dim: token dim
        depth: Attention block numbers
        heads: heads in multi-head attn
        attn_dropout: dropout in attn
        ff_dropout: drop in ff in attn
        use_cls_token: use cls token in FT-transformer but autoint it should be False
        groups: used in Memory block --> how many cluster prompts
        sum_num_per_group: used in Memory block --> topk to sum in each sum cluster prompts
        prod_num_per_group: used in Memory block --> topk to sum in each prod cluster prompts
        cluster: if True, prompt --> q, False, x --> q
        target_mode: if None, prompt --> q, if mix, [prompt, x] --> q
        token_num: how many token in the input x
        token_descent: use in MUCH-TOKEN dataset
        use_prod: use prod block
        num_special_tokens: =2
        categories: how many different cate in each cate ol
        out: =1 if regressioin else =cls number
        self.num_cont: how many cont col
        num_cont = args.num_cont
        num_cate: how many cate col
        '''
        dim = args.dim
        depth = args.depth
        heads = args.heads
        attn_dropout = args.attn_dropout
        ff_dropout = args.ff_dropout
        self.use_cls_token = args.use_cls_token
        groups = args.groups
        sum_num_per_group = args.sum_num_per_group
        prod_num_per_group = args.prod_num_per_group
        cluster = args.cluster
        target_mode = args.target_mode
        token_num = args.num_cont + args.num_cate
        token_descent = args.token_descent
        use_prod = args.use_prod
        num_special_tokens = args.num_special_tokens
        categories = args.categories
        out = args.out
        self.out = out
        self.num_cont = args.num_cont
        num_cont = args.num_cont
        num_cate = args.num_cate
        self.use_sigmoid = args.use_sigmoid
        qk_relu = args.qk_relu

        self.args = args
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_cont > 0, 'input shape must not be null'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        total_tokens = self.num_unique_categories + num_special_tokens + 1

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(args.categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        if self.num_cont > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_cont)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_cls_token=self.use_cls_token,
            groups=groups,
            sum_num_per_group=sum_num_per_group,
            prod_num_per_group=prod_num_per_group,
            cluster=cluster,
            target_mode=target_mode,
            token_num=token_num,
            token_descent=token_descent,
            use_prod=use_prod,
            qk_relu=qk_relu,
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, out)
        )

        self.pool = nn.Linear(num_cont + num_cate, 1)

    def model_name(self):
        return 'ft_trans'

    def forward(self, x_categ, x_numer, label, step=0, return_attn=False):
        # assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            # x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_cont > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim=1)

        # append cls tokens
        b = x.shape[0]

        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        # attend

        x = self.transformer(x)

        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = self.pool(x.transpose(-1, -2)).squeeze(-1)

        logit = self.to_logits(x)
        if self.out == 1:
            if self.use_sigmoid:
                logit = torch.sigmoid(logit)
            loss = nn.MSELoss()(logit.reshape(-1), label.float())
        else:
            loss = nn.CrossEntropyLoss()(logit, label)

        return logit, loss
    

class ArithmeticBlock(nn.Module):
    """
    AMFormer 
    """
    def __init__(self, embed_dim, num_prompts=64, top_k=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts
        self.top_k = top_k
        
        # 1. Prompt Tokens (Query)
        self.prompt_add = nn.Parameter(torch.randn(1, num_prompts, embed_dim))
        self.prompt_mul = nn.Parameter(torch.randn(1, num_prompts, embed_dim))
        
        # 2. Key and Value
        self.k_proj_add = nn.Linear(embed_dim, embed_dim)
        self.v_proj_add = nn.Linear(embed_dim, embed_dim)
        
        self.k_proj_mul = nn.Linear(embed_dim, embed_dim)
        self.v_proj_mul = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (Batch, Num_features, Embed_dim)
        B, N, D = x.shape
        
        # ==========================================
        # Additive Stream
        # ==========================================
        q_add = self.prompt_add.expand(B, -1, -1)  # (B, P, D)
        k_add = self.k_proj_add(x)                 # (B, N, D)
        v_add = self.v_proj_add(x)                 # (B, N, D)
        
        # Attention Scores: Q * K^T / sqrt(D)
        attn_add = torch.bmm(q_add, k_add.transpose(1, 2)) / math.sqrt(self.embed_dim) # (B, P, N)
        
        # Top-K Hard Attention
        if self.top_k < N:
            _, topk_idx = torch.topk(attn_add, self.top_k, dim=-1)
            mask = torch.zeros_like(attn_add).scatter_(-1, topk_idx, 1.0)
            attn_add = attn_add.masked_fill(mask == 0, float('-inf'))
            
        attn_add_weights = F.softmax(attn_add, dim=-1)
        out_add = torch.bmm(attn_add_weights, v_add) # (B, P, D)
        
        # ==========================================
        # Multiplicative Stream
        # ==========================================
        eps = 1e-5
        # log(ReLU(X) + eps)
        x_log = torch.log(F.relu(x) + eps)
        
        q_mul = self.prompt_mul.expand(B, -1, -1)
        k_mul = self.k_proj_mul(x_log)
        v_mul = self.v_proj_mul(x_log)
        
        attn_mul = torch.bmm(q_mul, k_mul.transpose(1, 2)) / math.sqrt(self.embed_dim)
        
        # Top-K Hard Attention
        if self.top_k < N:
            _, topk_idx = torch.topk(attn_mul, self.top_k, dim=-1)
            mask = torch.zeros_like(attn_mul).scatter_(-1, topk_idx, 1.0)
            attn_mul = attn_mul.masked_fill(mask == 0, float('-inf'))
            
        attn_mul_weights = F.softmax(attn_mul, dim=-1)
        out_mul_log = torch.bmm(attn_mul_weights, v_mul) # (B, P, D)
        
        out_mul = torch.exp(torch.clamp(out_mul_log, max=10.0))
        
        out_concat = torch.cat([out_add, out_mul], dim=1)
        return out_concat


class TrainAMFormer(nn.Module):
    def __init__(self, num_features, embed_dim, num_prompts=4, top_k=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_tokenizer = nn.Linear(num_features, embed_dim * num_features) 
        
        self.am_block = ArithmeticBlock(embed_dim, num_prompts, top_k)
        
        self.ffn_add = nn.Sequential(
            nn.Linear(embed_dim * num_prompts * 2, num_features),
            nn.Tanhshrink(),
            nn.LayerNorm(num_features),
        )
        
        width = 1024
        self.mlp = nn.Sequential(
            nn.Linear(num_features, width),
            nn.Tanhshrink(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.Tanhshrink(),
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )

    def forward(self, x, t0=10):
        B, N = x.shape
        x_emb = self.feature_tokenizer(x).view(B, N, self.embed_dim)
        interact_out = self.am_block(x_emb) 
        flat_out = self.ffn_add(interact_out.view(B, -1))
        log_hz = self.mlp(x + flat_out * 0.01)
        return log_hz


class InferAMFormer(nn.Module):
    def __init__(self, num_features, num_zscore, embed_dim, num_prompts=4, top_k=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_tokenizer = nn.Linear(num_features, embed_dim * num_features) 
        
        self.am_block = ArithmeticBlock(embed_dim, num_prompts, top_k)
        
        self.ffn_add = nn.Sequential(
            nn.Linear(embed_dim * num_prompts * 2, num_features),
            nn.Tanhshrink(),
            nn.LayerNorm(num_features),
        )
        
        width = 1024
        self.mlp = nn.Sequential(
            nn.Linear(num_features, width),
            nn.Tanhshrink(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.Tanhshrink(),
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )
        
        self.num_features = num_features
        self.num_zscore = num_zscore
        
        self.register_buffer("mean", torch.zeros(num_zscore, dtype=torch.float64))
        self.register_buffer("std", torch.ones(num_zscore, dtype=torch.float64))
        self.register_buffer("base_event_times", torch.zeros(101, dtype=torch.float64))
        self.register_buffer("base_hazards", torch.arange(0, 10.1, 0.1, dtype=torch.float64))

    def forward(self, x, t0=None):
        B, N = x.shape
        x[:, :self.num_zscore] = (x[:, :self.num_zscore] - self.mean) / self.std

        x_emb = self.feature_tokenizer(x).view(B, N, self.embed_dim)
        interact_out = self.am_block(x_emb)
        flat_out = self.ffn_add(interact_out.view(B, -1))
        log_hz = self.mlp(x + flat_out * 0.01)

        probs = 1.0 - torch.exp(- self.base_hazards * torch.exp(log_hz))

        if t0 is None:
            t0 = torch.tensor([10.0], device=x.device, dtype=self.base_event_times.dtype)
        elif not torch.is_tensor(t0):
            t0 = torch.tensor([t0], device=x.device, dtype=self.base_event_times.dtype)
        else:
            t0 = t0.to(device=x.device, dtype=self.base_event_times.dtype)

        t0 = t0.reshape(-1)
        if t0.numel() == 1:
            t0 = t0.expand(B)
        elif t0.numel() != B:
            raise ValueError(f"t0 size must be 1, got {t0.numel()}")

        idx = torch.searchsorted(self.base_event_times, t0, right=True) - 1
        idx = torch.clamp(idx, min=0, max=self.base_event_times.numel() - 1)

        batch_idx = torch.arange(B, device=x.device)
        return probs[batch_idx, idx]
