import torch 
from torch import nn

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