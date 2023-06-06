import numpy as np
import torch

# # FB15K-237数据集的罕见节点
rare_entity_list = ['/m/0gslw', '/m/01x4x4', '/m/01lk31', '/m/04z288', '/m/0jq47', '/m/09cmq', '/m/0dng4', '/m/05qc_', '/m/08hbxv', '/m/05hyf', '/m/0qb7t', '/m/027752', '/m/03f_jk', '/m/027qb1', '/m/01x8f6', '/m/025x7g_', '/m/080v2', '/m/037vqt', '/m/026y05', '/m/05k8m5', '/m/0hsph', '/m/02rxd26', '/m/07hn5', '/m/03tp4', '/m/04gbl3', '/m/01664_', '/m/0171c7', '/m/0zq7r', '/m/047vnfs', '/m/022qqh', '/m/0jvq', '/m/0lyb_', '/m/027yjnv', '/m/024030', '/m/024tv_', '/m/0q96']
# 自己创建的罕见节点
# rare_entity_list = ['1111', '2222']

entity_vec_dic = {}
relation_vec_dic = {}

path = "data/TransE-result/"
path = "data/TransGAT-result/"

with open(path + "entity_embedding.txt") as r1:         # 实体向量字典
    entitys = r1.readlines()
    for entity in entitys:
        entity = entity.strip('\n').split('\t')
        entity_vec_dic[entity[0]] = torch.tensor(eval(entity[1]))
with open(path + "relation_embedding.txt") as r2:         # 实体向量字典
    relations = r2.readlines()
    for relation in relations:
        relation = relation.strip('\n').split('\t')
        relation_vec_dic[relation[0]] = torch.tensor(eval(relation[1]))
# print(entity_vec_dic, relation_vec_dic)

with open("data/TransE-result/test.txt") as r3:     # 读train数据集
    triples = r3.readlines()
    for triple in triples:                      # 遍历三元组
        triple = triple.strip('\n').split('\t')     # 分割三元组
        if triple[0] in rare_entity_list or triple[2] in rare_entity_list:  # 筛选罕见三元组
            if triple[0] in rare_entity_list:
                print('entity : ', triple[0])
            elif triple[2]in rare_entity_list:
                print('entity : ', triple[2])
            h = triple[0]
            r = triple[1]
            t = triple[2]
            real_score = entity_vec_dic[h] + relation_vec_dic[r] - entity_vec_dic[t]
            real_score = real_score.numpy().sum()
            score_list = []
            for t in entity_vec_dic:        # 遍历所有实体
                score = entity_vec_dic[h] + relation_vec_dic[r] - entity_vec_dic[t]
                score = score.numpy().sum().tolist()
                score_list.append(score)
            score_list = sorted(score_list)
            rank = score_list.index(real_score)
            print('rank : ', rank)

