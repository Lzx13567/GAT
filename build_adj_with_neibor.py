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
       tmp_matrix = -9e15 * torch.ones(N, M, 3)  # 存放每行元素最大的M个值及坐标

       attention = torch.where(adj > 0, e, zero_vec)
       tmp_e = torch.where(adj > 0, zero_vec, e)  # 防止聚合e时取到邻居
       tmp_attention = attention

       # print(torch.max(attention, dim=1))

       # 无论邻居是否够M个，都挑选最大的M个，不足M个的就将负无穷放入tmp_matrix
       for i in range(M):
              tmp_matrix[:, i, 0] = torch.max(tmp_attention, dim=1).values  # 将最大值赋给tmp_matrix
              tmp_matrix[:, i, 1] = torch.tensor(np.array(range(N)))
              tmp_matrix[:, i, 2] = torch.max(tmp_attention, dim=1).indices  # 储存索引
              ind = (torch.LongTensor(tmp_matrix[:, i, 1].long()), torch.LongTensor(tmp_matrix[:, i, 2].long()))  # 最大值索引
              c = -9e15 * torch.ones(N)
              # c = torch.DoubleTensor(-9e15 * torch.ones(N).double())  # 赋0值
              tmp_attention = tmp_attention.index_put(ind, c)  # 将已经拿出来的最大值位置赋值0

       # print(tmp_matrix)

       # 把邻居不足M个而得到负无穷处的值替换为e中的值
       for i in range(N):
              if i % 1000 == 0:
                     print(i)
              sum_num = int(adj.sum(dim=1)[i].item())
              if sum_num < M:
                     for j in range(M - sum_num):
                            tmp_matrix[i, sum_num + j, 0] = tmp_e.max(dim=1).values[i].item()  # 将e中最大值放入tmp_matrix
                            tmp_matrix[i, sum_num + j, 2] = tmp_e.max(dim=1).indices[i].item()  # 将e中最大值索引放入tmp_matrix
                            ind = (torch.LongTensor(tmp_matrix[i, sum_num + j, 1].long()),
                                   torch.LongTensor(tmp_matrix[i, sum_num + j, 2].long()))  # 最大值索引
                            c = torch.tensor(-9e15)
                            # c = torch.DoubleTensor(torch.tensor(-9e15).double())  # 赋0值
                            tmp_e = tmp_e.index_put(ind, c)  # 将已经拿出来的最大值位置赋值0

       # print(tmp_matrix)

       # a = torch.squeeze(tmp_matrix[:, :, 1].reshape(-1, M * N), 0).detach().numpy().tolist()
       # b = torch.squeeze(tmp_matrix[:, :, 2].reshape(-1, M * N), 0).detach().numpy().tolist()
       # indices = torch.tensor([a, b])
       indices = torch.tensor([torch.squeeze(tmp_matrix[:, :, 1].reshape(-1, M * N), 0).detach().numpy().tolist(),
                               torch.squeeze(tmp_matrix[:, :, 2].reshape(-1, M * N), 0).detach().numpy().tolist()])
       values = torch.tensor(torch.squeeze(tmp_matrix[:, :, 0].reshape(-1, M * N), 0).detach().numpy().tolist(),
                             dtype=torch.float32)
       adj = torch.sparse_coo_tensor(indices=indices, values=values, size=[N, N])

       # print(adj.to_dense())

       # print('build_adj_with_neibor end.')

       return adj.to_dense()

#
# N = 10        # 实体数量
# M = 3         # 邻居数
#
# e = np.random.rand(N, N)
# adj = [[0,1,0,0,0,1,1,1,1,0],[0,0,1,0,0,0,0,0,0,0],[0,1,0,1,1,0,0,1,1,1],[0,0,1,0,1,0,0,0,0,0],[1,0,0,0,0,1,0,0,0,0],
#        [0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,1,1,0]]
# adj = np.array(adj)
#
# e = torch.tensor(e)
# adj = torch.tensor(adj)
# zero_vec = -9e15 * torch.ones_like(e)
# tmp_matrix = -9e15 * torch.ones(N, M, 3)         # 存放每行元素最大的M个值及坐标
# tmp_e = e                                        # e的副本
# attention = torch.where(adj > 0, e, zero_vec)
# tmp_e = torch.where(adj > 0, zero_vec, e)        # 防止聚合e时取到邻居
# tmp_attention = attention
#
# # print(torch.max(attention, dim=1))
#
# # 无论邻居是否够M个，都挑选最大的M个，不足M个的就将负无穷放入tmp_matrix
# for i in range(M):
#        tmp_matrix[:, i, 0] = torch.max(tmp_attention, dim=1).values              # 将最大值赋给tmp_matrix
#        tmp_matrix[:, i, 1] = torch.tensor(np.array(range(N)))
#        tmp_matrix[:, i, 2] = torch.max(tmp_attention, dim=1).indices             # 储存索引
#        ind = (torch.LongTensor(tmp_matrix[:, i, 1].long()), torch.LongTensor(tmp_matrix[:, i, 2].long()))       # 最大值索引
#        c = torch.DoubleTensor(-9e15 * torch.ones(N).double())                           # 赋0值
#        tmp_attention = tmp_attention.index_put(ind, c)                           # 将已经拿出来的最大值位置赋值0
#
# # print(tmp_matrix)
#
# # 把邻居不足M个而得到负无穷处的值替换为e中的值
# for i in range(N):
#        if adj.sum(dim=1)[i].item() < M:
#               for j in range(M - adj.sum(dim=1)[i].item()):
#                      tmp_matrix[i, adj.sum(dim=1)[i].item() + j, 0] = tmp_e.max(dim=1).values[i].item()      # 将e中最大值放入tmp_matrix
#                      tmp_matrix[i, adj.sum(dim=1)[i].item() + j, 2] = tmp_e.max(dim=1).indices[i].item()     # 将e中最大值索引放入tmp_matrix
#                      ind = (torch.LongTensor(tmp_matrix[i, adj.sum(dim=1)[i].item() + j, 1].long()),
#                             torch.LongTensor(tmp_matrix[i, adj.sum(dim=1)[i].item() + j, 2].long()))  # 最大值索引
#                      c = torch.DoubleTensor(torch.tensor(-9e15).double())  # 赋0值
#                      tmp_e = tmp_e.index_put(ind, c)  # 将已经拿出来的最大值位置赋值0
#
# # print(tmp_matrix)
#
# a = torch.squeeze(tmp_matrix[:, :, 1].reshape(-1, M * N), 0).numpy().tolist()
# b = torch.squeeze(tmp_matrix[:, :, 2].reshape(-1, M * N), 0).numpy().tolist()
# indices = torch.tensor([a, b])
# values = torch.tensor(torch.squeeze(tmp_matrix[:, :, 0].reshape(-1, M * N), 0).numpy().tolist(), dtype=torch.float32)
# adj = torch.sparse_coo_tensor(indices=indices, values=values, size=[N, N])
#
# print(adj.to_dense())
