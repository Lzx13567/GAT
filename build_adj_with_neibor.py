import numpy as np
import torch
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from build_matrix import get_value
import numpy as np


def build_adj_with_neibor (M, N, e, adj):

       print('build_adj_with_neibor is running.')

       zero_vec = -9e15 * torch.ones_like(e)
       tmp_matrix = -9e15 * torch.ones(N, M, 3)  

       attention = torch.where(adj > 0, e, zero_vec)
       tmp_e = torch.where(adj > 0, zero_vec, e) 
       tmp_attention = attention

       for i in range(M):
              tmp_matrix[:, i, 0] = torch.max(tmp_attention, dim=1).values
              tmp_matrix[:, i, 1] = torch.tensor(np.array(range(N)))
              tmp_matrix[:, i, 2] = torch.max(tmp_attention, dim=1).indices
              ind = (torch.LongTensor(tmp_matrix[:, i, 1].long()), torch.LongTensor(tmp_matrix[:, i, 2].long())) 
              c = -9e15 * torch.ones(N)
              # c = torch.DoubleTensor(-9e15 * torch.ones(N).double())  
              tmp_attention = tmp_attention.index_put(ind, c) 
              
       for i in range(N):
              if i % 1000 == 0:
                     print(i)
              sum_num = int(adj.sum(dim=1)[i].item())
              if sum_num < M:
                     for j in range(M - sum_num):
                            tmp_matrix[i, sum_num + j, 0] = tmp_e.max(dim=1).values[i].item() 
                            tmp_matrix[i, sum_num + j, 2] = tmp_e.max(dim=1).indices[i].item()
                            ind = (torch.LongTensor(tmp_matrix[i, sum_num + j, 1].long()),
                                   torch.LongTensor(tmp_matrix[i, sum_num + j, 2].long())) 
                            c = torch.tensor(-9e15)
                            # c = torch.DoubleTensor(torch.tensor(-9e15).double()) 
                            tmp_e = tmp_e.index_put(ind, c)  

       indices = torch.tensor([torch.squeeze(tmp_matrix[:, :, 1].reshape(-1, M * N), 0).detach().numpy().tolist(),
                               torch.squeeze(tmp_matrix[:, :, 2].reshape(-1, M * N), 0).detach().numpy().tolist()])
       values = torch.tensor(torch.squeeze(tmp_matrix[:, :, 0].reshape(-1, M * N), 0).detach().numpy().tolist(),
                             dtype=torch.float32)
       adj = torch.sparse_coo_tensor(indices=indices, values=values, size=[N, N])
       return adj.to_dense()
