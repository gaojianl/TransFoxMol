import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv
from torch_geometric.utils import to_dense_batch
from math import sqrt
from TFM.utils import get_attn_pad_mask, create_ffn


class Embed(nn.Module):
    def __init__(self, attn_head=4, output_dim=128, d_k=64, d_v=64, attn_layers=4, dropout=0.1, disw=1.5, device='cuda:0'):
        super(Embed, self).__init__()
        self.device = device 
        self.relu = nn.ReLU()         
        self.disw = disw
        self.layer_num = attn_layers 
        self.gnns = nn.ModuleList([GraphConv(36, output_dim) if i == 0 else GraphConv(output_dim, output_dim) for i in range(attn_layers)])
        self.nms = nn.ModuleList([nn.LayerNorm(output_dim) for _ in range(attn_layers)])
        self.dps = nn.ModuleList([nn.Dropout(dropout) for _ in range(attn_layers)])
        self.tfs = nn.ModuleList([Encoder(output_dim, d_k, d_v, 1, attn_head, dropout) for _ in range(attn_layers)])

    def forward(self, x, edge_index, edge_attr, batch, leng, adj, dis):
        x = self.gnns[0](x, edge_index, edge_weight=edge_attr)
        x = self.dps[0](self.nms[0](x))
        x = self.relu(x)

        x_batch, mask = to_dense_batch(x, batch)

        batch_size, max_len, output_dim = x_batch.size()
        matrix_pad = torch.zeros((batch_size, max_len, max_len))
        for i, l in enumerate(leng):
            adj_ = torch.FloatTensor(adj[i]); dis_ = torch.FloatTensor(dis[i])
            dis_ = 1 / torch.pow(self.disw, (dis_ - 1))
            dis_ = torch.where(dis_ == self.disw, torch.zeros_like(dis_), dis_)
            matrix = torch.where(adj_ == 0, dis_, adj_)
            matrix_pad[i, :int(l[0]), :int(l[0])] = matrix
        matrix_pad = matrix_pad.to(self.device)

        x_batch = self.tfs[0](x_batch, mask, matrix_pad)
        for i in range(1, self.layer_num):
            x = torch.masked_select(x_batch, mask.unsqueeze(-1))
            x = x.reshape(-1, output_dim)
            x = self.gnns[i](x, edge_index, edge_weight=edge_attr)
            x = self.dps[i](self.nms[i](x))

            x = self.relu(x)
            x_batch, mask = to_dense_batch(x, batch)
            x_batch = self.tfs[i](x_batch, mask, matrix_pad)

        return x_batch


class Fox(nn.Module):
    def __init__(self, task='reg', tasks=1, attn_head=4, output_dim=128, d_k=64, d_v=64, attn_layers=4, D=16, dropout=0.1, disw=1.5, device='cuda:0'):
        super(Fox, self).__init__()                                                                                                                    
        self.device = device
        self.emb = Embed(attn_head, output_dim, d_k, d_v, attn_layers, dropout, disw, device)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(D, output_dim))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(2, D))
        self.th = nn.Tanh()
        self.sm = nn.Softmax(-1)
        self.bm = nn.BatchNorm1d(2, output_dim)

        self.act = create_ffn(task, tasks, output_dim, dropout)
        self.reset_params()

    def reset_params(self):
        for weight in self.parameters():
            if len(weight.size()) > 1:
                nn.init.xavier_normal_(weight)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device)                                               # tensor
        leng, adj, dis = data.leng, data.adj, data.dis
        batch = data.batch.to(self.device)
        
        x_batch = self.emb(x, edge_index, edge_attr, batch, leng, adj, dis)

        x_bat = self.th(torch.matmul(self.w1, x_batch.permute(0, 2, 1)))  
        x_bat = self.sm(torch.matmul(self.w2, x_bat))          
        x_p = self.bm(torch.matmul(x_bat, x_batch))
        x_p = x_p.reshape(x_p.size(0), x_p.size(1)*x_p.size(2))

        # prediction
        logits = self.act(x_p)

        return logits


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dp = nn.Dropout(dropout)
        self.sm = nn.Softmax(dim=-1)
        
    def forward(self, Q, K, V, attn_mask, matrix):
        scores_ = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(self.d_k)  
        scores = scores_*matrix
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.sm(scores)
        context = torch.matmul(self.dp(attn), V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.fc = nn.Linear(d_v*n_heads, d_model, bias=False)
        self.nm = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dp = nn.Dropout(p=dropout)
        self.sdpa = ScaledDotProductAttention(d_k, dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask, matrix):
        batch_size = input_Q.size(0)
        
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) 
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)      

        if len(matrix.size()) == 3:
            matrix = matrix.unsqueeze(1).repeat(1, self.n_heads, 1, 1)    

        context = self.sdpa(Q, K, V, attn_mask, matrix)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)                                        
        
        return self.dp(self.nm(output))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.nm = nn.LayerNorm(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask, matrix):
        residual = enc_inputs
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, matrix)
        return self.nm(enc_outputs+residual)


class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_layers, n_heads, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, mask, matrix):
        enc_self_attn_mask = get_attn_pad_mask(mask)
        for layer in self.layers:
            enc_inputs = layer(enc_inputs, enc_self_attn_mask, matrix)
        return enc_inputs