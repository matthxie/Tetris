import torch 
import torch.nn as nn
import numpy as np

class ViT(nn.Module):
    def __init__(self, n_heads, n_enc, n_dec, n_blocks=8, n_tokens=8):
        super(ViT, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.n_tokens = n_tokens
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.block_size = 5

        self.linear_map = nn.Linear(self.n_blocks*self.block_size**2, self.n_tokens)
        self.classifier = nn.Parameter(torch.rand(1, self.n_tokens))

        self.pos_emdding = nn.Parameter(torch.tensor(self.create_positional_encoding(
            self.n_blocks ** 2 + 1, self.n_tokens
        )))
        self.pos_emdding.requires_grad = False

        self.enc_blocks = nn.ModuleList([
            ViTBlock(n_tokens, n_heads) for _ in range(n_enc)
        ])

    def forward(self, input):
        blocks = self.create_blocks(input).to(device)

        tokens = self.linear_map(blocks.type(torch.float32))
        tokens = self.classifier([torch.vstack((self.classifier, tokens[i])) \
                                 for i in range(len(tokens))])
        pos_embeddings = self.pos_emdding.repeat(self.n_tokens, 1, 1)
        logits = tokens + pos_embeddings

        for enc_block in self.enc_blocks:
            logits = enc_block(logits)

        return logits

    def create_blocks(self, input):
        height = input.size()[0]
        width = input.size()[1]
        # blocks = torch.zeros(self.n_blocks*self.block_size**2).to(self.device)
        blocks = torch.zeros(self.block_size**2).to(self.device)
        # blocks = torch.tensor([]).to(self.device)

        print(input.shape)
        # print(blocks.shape)

        for i in range(height//self.block_size):
            for j in range(width//self.block_size):
                block = input[i*self.block_size : (i + 1)*self.block_size, \
                              j*self.block_size : (j+1)*self.block_size]
                torch.hstack((blocks, block.flatten()))
                # blocks[i*self.block_size+j*self.block_size] = block.flatten()

        print(blocks.shape)
        return blocks
    
    def create_positional_encoding(self, length, dim):
        result = torch.ones(length, dim)
        for i in range(length):
            for j in range(dim):
                if j%2 == 0:
                    result[i][j] = np.sin(i / (10000 ** (j/dim)))
                else:
                    result[i][j] == np.cos(i / (10000 ** ((j-1)/dim)))
        return result
    
class ViTBlock(nn.Module):
    def __init__(self, n_tokens, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(n_tokens)
        self.msa = MSA(n_tokens, n_heads)
        self.norm2 = nn.LayerNorm(n_tokens)
        self.mlp = nn.Sequential(
            nn.Linear(n_tokens, mlp_ratio*n_tokens),
            nn.GELU(),
            nn.Linear(mlp_ratio*n_tokens, n_tokens)
        )

    def forward(self, input):
        out = input + self.msa(self.norm1(input))
        out = out + self.mlp(self.norm2(out))
        return out
    
class MSA(nn.Module):
    def __init__(self, dim, n_heads=2):
        super(MSA, self).__init__()

        self.dim = dim
        self.n_heads = n_heads

        d_head = int(dim/n_heads)
        self.q_proj = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_heads)])
        self.v_proj = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                query = self.q_proj[head]
                key = self.k_proj[head]
                value = self.v_proj[head]

                seq = sequence[:, head*self.d_head : (head+1)*self.d_head]
                q, k, v  = query(seq), key(seq), value(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)

            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViT(2, 2, 0, 8, 8).to(device)
x = torch.randn((10, 20)).to(device)

print(model((x)))