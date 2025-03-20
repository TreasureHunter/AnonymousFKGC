import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import networkx as nx

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['Nell-One','Wiki-One','Fb15k-237'])
parser.add_argument('--eta', type=float, default=0.03)
args = parser.parse_args()
data_path=f'./{args.dataset}'
rel2id = json.load(open(data_path + '/relation2ids'))
ent2id = json.load(open(data_path + '/ent2ids'))
ent_embed = np.loadtxt(data_path + '/emb/entity2vec.TransE')
rel_embed = np.loadtxt(data_path + '/emb/relation2vec.TransE')
i = 0
embeddings = []
update_entity_embeddings =[[] for _ in range(len(ent2id.keys()))]  
symbol_id={}
embed_ent_id={}
e1_rele2 = defaultdict(list)
e1_degrees = defaultdict(int)
for key in rel2id.keys():
    if key not in ['','OOV']:
        symbol_id[key] = i
        i += 1
        embeddings.append(list(rel_embed[rel2id[key],:]))
for key in ent2id.keys():
    if key not in ['', 'OOV']:
        symbol_id[key] = i
        embed_ent_id[i] = ent2id[key]
        i += 1
        embeddings.append(list(ent_embed[ent2id[key],:]))
symbol_id['PAD'] = i
embeddings.append(list(np.zeros((rel_embed.shape[1],))))
embeddings = np.array(embeddings)
pad_id = len(symbol_id.keys())

with open(data_path + '/path_graph') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        e1,rel,e2 = line.rstrip().split()
        e1_rele2[e1].append((symbol_id[e1], symbol_id[rel], symbol_id[e2]))
        e1_rele2[e2].append((symbol_id[e2], symbol_id[rel+'_inv'], symbol_id[e1]))

degrees = {}
connections=[]
for ent, id_ in ent2id.items():
    neighbors = e1_rele2[ent]
    degrees[ent] = len(neighbors)
    e1_degrees[id_] = len(neighbors)
    temp_array = (np.ones((1, len(neighbors), 3)) * pad_id).astype(int)
    for idx, _ in enumerate(neighbors):
        temp_array[0, idx, 0] = _[0]
        temp_array[0, idx, 1] = _[1]
        temp_array[0, idx, 2] = _[2]
    connections.append(temp_array)
# connections = np.concatenate(connections)

for i in range(len(ent2id.keys())):
    temp_ent=connections[i]
    neighbor_id=temp_ent[0, :, 2].reshape(-1).tolist()
    id_= embed_ent_id[i+len(rel2id.keys())]
    if len(neighbor_id)==1:
        update_entity_embeddings[id_]=list(embeddings[len(rel2id.keys())+i])
        continue
    degree_list=[min(e1_degrees[embed_ent_id[nid]],3) for nid in neighbor_id]
    degree_sum=sum(degree_list)
    temp_embedding=0
    for nid in neighbor_id:
        temp_embedding += (min(e1_degrees[embed_ent_id[nid]],3)/degree_sum)*ent_embed[embed_ent_id[nid]]
    temp_embedding=temp_embedding*args.eta + embeddings[len(rel2id.keys())+i]*(1-args.eta)
    update_entity_embeddings[id_]=list(temp_embedding)
update_entity_embeddings = np.array(update_entity_embeddings)
print(update_entity_embeddings.shape)
print(len(ent2id.keys()))
num=int(args.eta*100)
save_file = data_path + f'/emb/entity2vec_degree{num}.TransE'
np.savetxt(save_file, update_entity_embeddings, fmt='%.6f', delimiter='\t')
