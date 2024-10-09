import torch 
from torch import nn
import math

#training parameters
B = 32 #batch size

#image parameters
C = 3
H = 128
W = 128
x = torch.rand(B, C, H, W)

#model parameters
D = 64 #hidden size
P = 4 #patch size
N = int(H*W/P**2) #number of tokens
k = 4 #number of attention heads
Dh = int(D/k) #attention head size
p = 0.1 #dropout rate
mlp_size = D*4
L = 4 #number of transformer blocks
n_classes = 3 #number of classes

# Image Embeddings [Patch, Class, with Possition Embeddings]

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()

        self.unfold = nn.Unfold(kernel_size = P, stride = P) #function to create patch vectors (x_p^i)
        self.project = nn.Linear(P**2*C, D) #patch tokens (E)
        self.cls_token = nn.Parameter(torch.randn((1, 1, D))) #function to create unbatched class token (x_class) as trainable parameter
        self.pos_embedding = nn.Parameter(torch.randn((1, N+1, D))) #function to create unbatched position embedding (E_pos) as trainable parameter
        self.dropout = nn.Dropout(p) 

        #Why unbatched? Because it allows to set parameters and functions here
        #Batched will increase parameter size without effectively improving it

    def forward(self, x, verbose = False): 
        print("######")
        print("input image:", x.shape)
        x = self.unfold(x).transpose(1, 2) #patch vectors (x_p^i)
        print("x_p^i:", x.shape)
        x = self.project(x)
        print("x_p^i*E:", x.shape) #tokens for patches (x_p^i*E)
        cls_token = self.cls_token.expand(B, -1, -1) #batched class token (x_class)
        print("x_class:", cls_token.shape)
        x = torch.cat((cls_token, x), dim = 1) #concatenate class, final image token embedding
        print("patch embedding:", x.shape)
        pos_embedding = self.pos_embedding.expand(B, -1, -1) #batched pos embedding (E_pos)
        print("E_pos:", pos_embedding.shape)
        z0 = x + pos_embedding #adding both embeddings
        print("z0:", z0.shape)
        z0 = self.dropout(z0) #dropout
        return z0



# Single Head Attention

class Single_Head_Attention(nn.Module):
    def __init__(self):
        super(Single_Head_Attention, self).__init__()

        self.U_qkv = nn.Linear(D, 3*Dh)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, z):
        print("z:", z.shape)
        qkv = self.U_qkv(z)
        print("qkv:", qkv.shape)
        q = qkv[:,:,:Dh]
        print("q:", q.shape)
        k = qkv[:,:,Dh:2*Dh]
        print("k:", k.shape)
        v = qkv[:,:,2*Dh:]
        print("v:", v.shape)
        qkTbysqrtDh = torch.matmul(q, k.transpose(-2, -2))/math.sqrt(Dh)
        print("qkTbysqrtDh:", qkTbysqrtDh.shape)
        A = self.softmax(qkTbysqrtDh)
        print("A:", A.shape)
        SAz = torch.matmul(A, v)
        print("SAz:", SAz)

        return SAz
    
# Multi Head Self Attention

class Multi_Head_Self_Attention(nn.Module):
    def __init__(self):
        super(Multi_Head_Self_Attention, self).__init__()

        self.heads = nn.ModuleList([Single_Head_Attention() for _ in range(k)])
        self.U_msa = nn.Linear(D, D)
        self.dropout = nn.Dropout(p)

    def forward(self, z):
        print("#####")
        print("z:", z.shape)
        ConSAz = torch.cat([head(z) for head in self.heads], dim = -1)
        print("ConSA(z):", ConSAz.shape)
        msaz = self.U_msa(z)
        print("MSA(z):", msaz.shape)
        msaz = self.dropout(msaz)

        return msaz
    

# MLP

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.U_mlp = nn.Linear(D, mlp_size)
        self.gelu = nn.GELU()
        self.U_mlp2 = nn.Linear(mlp_size, D)
        self.dropout = nn.Dropout(p)

    def forward(self, z):
        print("###MLP###")
        print("z:", z.shape)
        z = self.U_mlp(z)
        print("mlp(z):", z.shape)
        z = self.gelu(z)
        z = self.dropout(z)
        z = self.U_mlp2(z)
        print("mlp2(gelu(mlp(z))):", z.shape)
        z = self.gelu(z)
        z = self.dropout(z)

        return z
    

# Transformer Block

class Transformer_Block(nn.Module):
    def __init__(self):
        super(Transformer_Block, self).__init__()

        self.layernorm_1 = nn.LayerNorm(D)
        self.msa = Multi_Head_Self_Attention()
        self.layernorm_2 = nn.LayerNorm(D)
        self.mlp = MLP()

    def forward(self, z):
        print("###Transformer Block###")
        print("z:", z.shape)
        z1 = self.layernorm_1(z)
        print("layernorm_1:", z1.shape)
        z1 = self.msa(z1)
        print("msa(layernorm_1(z)):", z1.shape)
        z2 = z + z1
        print("z + msa(layernorm_1(z)):", z2.shape)
        z3 = self.layernorm_2(z2)
        print("layernorm_2(z + msa(layernorm_1(z))):", z3.shape)
        z3 = self.mlp(z3)
        print("mlp(layernorm_2(z + msa(layernorm_1(z)))):", z3.shape)
        z4 = z2 + z3
        print("z2 + mlp(layernorm_2(z + msa(layernorm_1(z)))):", z4.shape)

        return z4
    


# ViT

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()

        self.embedding = Embedding()
        self.transformer_encoder = nn.ModuleList([Transformer_Block() for _ in range(L)])
        self.layernorm = nn.LayerNorm(D)
        self.U_mlp = nn.Linear(D, n_classes)

    def forward(self, x):
        print("###ViT###")
        print("input image:", x.shape)
        z = self.embedding(x)
        print("z:", z.shape)
        for block in self.transformer_encoder:
            z = block(z)
        print("z:", z.shape)
        z = self.layernorm(z)
        print("layernorm(z):", z.shape)
        z = z[:,0,:]
        print("z:", z.shape)
        z = self.U_mlp(z)
        print("mlp(layernorm(z)):", z.shape)

        return z


