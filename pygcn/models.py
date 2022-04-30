from pygcn.layers import GraphConvolution
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import os, math
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class TMDLayer(nn.Module):
    def __init__(
        self,
        in_features = 100,
        L_latent = 32,
        epsilon = 0.25,
        adj_initialize = False
    ):
        super().__init__()
        
        # what should be the dimension here? should this be user input?

        self.pi_list = nn.Sequential(nn.Linear(L_latent, in_features), 
                                                    nn.ReLU(),
                                                    nn.Linear(in_features, 1),
                                                    nn.Sigmoid())
        self.dt = nn.Parameter(torch.FloatTensor([0.1]))
        self.adj_initialize =  adj_initialize

        self.epsilon = epsilon
        self.proj_list = nn.Sequential(nn.Linear(in_features, L_latent))
    

    def TMD_map(self, x, adj):
        # input x if of size [B, N, d]

        # print(x.shape)
        
        n,c = x.shape
        x = x.view(1,n,-1)
    

        x = self.proj_list(x)
        # L = construct from pe

        i_minus_j = x.unsqueeze(2) - x.unsqueeze(1)
        # print(K_epsilon.shape,adj.unsqueeze(0).shape)
        #K_epsilon = torch.exp(-1 / (4 * self.epsilon) * (i_minus_j ** 2).sum(dim=3))
        K_epsilon= + torch.clone(adj)

        # if self.adj_initialize:
        #     K_epsilon = torch.clone(adj)
        # else:
        #     K_epsilon = torch.exp(-1 / (4 * self.epsilon) * (i_minus_j ** 2).sum(dim=3))

        ## construct TMD

        q_epsilon_tilde = K_epsilon.sum(dim=2)
        D_epsilon_tilde = torch.diag_embed(self.pi_list(x).squeeze(2) / q_epsilon_tilde)
        K_tilde = K_epsilon.bmm(D_epsilon_tilde)
        D_tilde = torch.diag_embed(K_tilde.sum(dim=2) +
                                   1e-5 * torch.ones(K_tilde.shape[0], K_tilde.shape[1]).to(x.device))
        L = 1 / self.epsilon * (torch.inverse(D_tilde).bmm(K_tilde)) - torch.eye(K_tilde.shape[1]).to(
            x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)

        return L



    def forward(self, x, adj, f):
        L = self.TMD_map(x, adj).squeeze(0)
        
        target = F.relu(f(x))
        
        shape_val = target.shape
                
        target = (target + self.dt*torch.matmul(L, target.view(shape_val[0],-1)).view(shape_val))

        return target
    
class MLayer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.tmd_layer1 = TMDLayer(in_features = nfeat, L_latent=nhid, adj_initialize= True)


        self.l1 = nn.Linear(nfeat, nhid)

        self.l2 = nn.Linear(nhid, nclass)

        self.dropout = dropout
        # self.tmd_layer2 = TMDLayer(in_features = nhid, L_latent=12)

    def forward(self, x, adj):
        adj = torch.spmm(adj, torch.eye(adj.shape[0]))

        # print(adj.shape)
        n1, c = adj.shape
        adj = adj.view(1, n1, c)

        x = self.tmd_layer1(x, adj, self.l1)
        # print(x.sum(dim=1))
        # x = F.relu(self.l1(x))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.l2(x)
        # x = self.tmd_layer2(x, adj, self.l2)
        
        return F.log_softmax(x, dim=1)
