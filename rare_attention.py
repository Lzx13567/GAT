import torch
import torch.nn as nn
import torch.nn.functional as F
from build_matrix import get_value
from build_adj import get_adj
from build_adj_with_neibor import build_adj_with_neibor
import numpy as np

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        h = h.to(torch.float32)
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        M = 10
        N = e.size()[0]


        print('build_adj_with_neibor is running.')

        zero_vec = -9e15 * torch.ones_like(e)
        global tmp_matrix
        tmp_matrix = -9e15 * torch.ones(N, M, 3) 

        attention = torch.where(adj > 0, e, zero_vec)
        tmp_e = torch.where(adj > 0, zero_vec, e)
        tmp_attention = attention
