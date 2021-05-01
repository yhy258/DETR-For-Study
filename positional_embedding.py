import torch
import torch.nn as nn
class PositionalEmbedding(nn.Module):
    def __init__(self, n_dim): # 실제 transformer에서 사용하는 embedding dimension
        super().__init__()
        self.embed_dim = int(n_dim/2)
        self.row_embed = nn.Embedding(50, self.embed_dim)
        self.col_embed = nn.Embedding(50, self.embed_dim)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h,w = x.size()[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_embed = self.col_embed(i) # w, embed_dim
        y_embed = self.row_embed(j) # h, embed_dim

        pos = torch.cat([
                            x_embed.unsqueeze(0).repeat(h,1,1),
                            y_embed.unsqueeze(1).repeat(1,w,1)
                            ], dim=-1).permute(2,0,1).unsqueeze(0).repeat(x.shape[0],1,1,1)
        return pos # batch_size
