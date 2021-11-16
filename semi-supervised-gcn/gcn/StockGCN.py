
import torch
import torch_geometric
import numpy as np
import pickle
import json
import pandas as pd
import networkx as nx
import os
import json
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

base_dir = './data/'

with open(os.path.join(base_dir, 'earning_calls_ticker_index.json'), 'rb') as f:
    ticker_indices = json.load(f)
indices_tickers = {}
for k in ticker_indices.keys():
    indices_tickers[ticker_indices[k]]=k

adj_wiki = pd.read_csv(os.path.join(base_dir, 'adj_wiki.csv'))
adj_wiki = adj_wiki.values
adj_wiki = adj_wiki[:,1:]
print('Shape of wiki-company based adjacenecy matrix is : ', adj_wiki.shape)

adj_graph = adj_wiki

with open('../../cross-modal-attn/audio_featDict.pkl', 'rb') as f:
    audio_featDict=pickle.load(f)
    
with open('../../cross-modal-attn/audio_featDictMark2.pkl', 'rb') as f:
    audio_featDictMark2=pickle.load(f)

#with open('./data/finbert_earnings.pkl', 'rb') as f:
#    text_dict=pickle.load(f)
#
#with open('./data/n2vmulti_embd.pkl', 'rb') as f:
#    graph_embd_dict=pickle.load(f)
    
traindf= pd.read_csv("./data/train_split3.csv")
testdf=pd.read_csv("./data/test_split3.csv")
valdf=pd.read_csv("./data/val_split3.csv")

call_dict = {} ## the index of the call in the adjacency matrix
call_stock_ind = {}
i=277
start = i
for index, row in traindf.iterrows():
    if not row['text_file_name'] in call_dict.keys():
        call_dict[row['text_file_name']] = i
        if row['ticker'].upper() in ticker_indices.keys():
            call_stock_ind[row['text_file_name']] = ticker_indices[row['ticker'].upper()]
        else:
            call_stock_ind[row['text_file_name']] = -1
        i+=1
train_end = i
train_idx = range(start, train_end)
for index, row in testdf.iterrows():
    if not row['text_file_name'] in call_dict.keys():
        call_dict[row['text_file_name']] = i
        if row['ticker'].upper() in ticker_indices.keys():
            call_stock_ind[row['text_file_name']] = ticker_indices[row['ticker'].upper()]
        else:
            call_stock_ind[row['text_file_name']] = -1
        i+=1
test_end = i
test_idx = range(train_end, test_end)
for index, row in valdf.iterrows():
    if not row['text_file_name'] in call_dict.keys():
        call_dict[row['text_file_name']] = i
        if row['ticker'].upper() in ticker_indices.keys():
            call_stock_ind[row['text_file_name']] = ticker_indices[row['ticker'].upper()]
        else:
            call_stock_ind[row['text_file_name']] = -1
        i+=1
val_end = i
val_idx = range(test_end, val_end)
print('Number of unique calls:', len(call_dict), len(call_stock_ind))
print('Total number of graph nodes:', i)
print('Number of train values', len(train_idx))
print('Number of test values', len(test_idx))
print('Number of validation values', len(val_idx))
new_adj_matrix_shape = (i, i)

final_adj = np.zeros(new_adj_matrix_shape,dtype=np.float32)
final_adj[:adj_graph.shape[0], :adj_graph.shape[1]]=adj_graph
## add calls to graph

for idx, row in traindf.iterrows():
    stock_name = row['text_file_name']
    final_adj[call_dict[stock_name], call_dict[stock_name]] = 1.0
    if call_stock_ind[stock_name]!= -1:
        final_adj[call_dict[stock_name], call_stock_ind[stock_name]] = 1.0
        final_adj[call_stock_ind[stock_name], call_dict[stock_name]] = 1.0

for idx, row in testdf.iterrows():
    stock_name = row['text_file_name']
    final_adj[call_dict[stock_name], call_dict[stock_name]] = 1.0
    if call_stock_ind[stock_name]!= -1:
        final_adj[call_dict[stock_name], call_stock_ind[stock_name]] = 1.0
        final_adj[call_stock_ind[stock_name], call_dict[stock_name]] = 1.0

for idx, row in valdf.iterrows():
    stock_name = row['text_file_name']
    final_adj[call_dict[stock_name], call_dict[stock_name]] = 1.0
    if call_stock_ind[stock_name]!= -1:
        final_adj[call_dict[stock_name], call_stock_ind[stock_name]] = 1.0
        final_adj[call_stock_ind[stock_name], call_dict[stock_name]] = 1.0
        
print('Final adjacency matrix shape: ', final_adj.shape)
print('Number of connections: ', np.sum(final_adj))

np.save('final_adj.npy', final_adj)
final_adj = np.load('final_adj.npy')

G = nx.from_numpy_matrix(final_adj)
print(nx.info(G))
nx.write_edgelist(G,"gcn.edgelist")
edgelist = pd.read_csv('gcn.edgelist',header=None)
print(edgelist.head())
edgelist.columns = ['edge']
edgelist  = edgelist.edge.str.split(" ",expand=True)
edgelist.columns = ['node1','node2','wt','wtval']
print(edgelist.head())
edges = []
for ind in edgelist.index:
    edges.append([edgelist['node2'][ind],edgelist['node1'][ind]])
print('Edge list of type: ', type(edges), ' and of length: ', len(edges))
edges_np = np.array(edges)
print('Shape of numpy array of edges: ', edges_np.shape)
edges_transpose = np.transpose(edges_np)
print('Shape of numpy array of edges as required by GCN: ', edges_transpose.shape)
np.save('edgelist.npy', edges_transpose)
edges_transpose = np.load('edgelist.npy')

def store_call_indices_for_stocks():
    stock_call_dict = {}
    for i in range(277):
        stock_call_dict[i] = []
    for call in call_stock_ind.keys():
        if call_stock_ind[call]!= -1:
            stock_call_dict[call_stock_ind[call]].append(call_dict[call])
    return stock_call_dict

stock_call_ind = store_call_indices_for_stocks()
indices_file_name = {}

not_present = np.ones(836)
for i in range(836):
    for key in call_dict.keys():
        if call_dict[key] == i:
            indices_file_name[i] = key
            not_present[i] =0
    for key in call_stock_ind.keys():
        if call_stock_ind[key]==i:
            indices_file_name[i] = key[:-9]
            not_present[i]=0

np.argwhere(not_present)

full_stock_data = pd.read_csv('./data/full_stock_data.csv')
#full_stock_data.head()

TOTAL_NODES = final_adj.shape[0]

def get_labels():
    y3days = np.zeros(TOTAL_NODES, dtype=np.float32)
    y7days = np.zeros(TOTAL_NODES, dtype=np.float32)
    y15days = np.zeros(TOTAL_NODES, dtype=np.float32)
    y30days = np.zeros(TOTAL_NODES, dtype=np.float32)
    for index, row in full_stock_data.iterrows():
        if row['text_file_name'] in call_dict.keys():
            y3days[call_dict[row['text_file_name']]]= row['future_3']
            y7days[call_dict[row['text_file_name']]]= row['future_7']
            y15days[call_dict[row['text_file_name']]]= row['future_15']
            y30days[call_dict[row['text_file_name']]]= row['future_30']
    return y3days, y7days, y15days, y30days

y3days, y7days, y15days, y30days = get_labels()
print(y3days.shape, y7days.shape, y15days.shape, y30days.shape)

stock_idx = range(277)

mask0 = np.zeros(y3days.shape[0])
mask0[list(stock_idx)] = 1
stock_mask = np.array(mask0, dtype=np.bool)
print(stock_mask.shape)

mask = np.zeros(y3days.shape[0])
mask[list(train_idx)] = 1
train_mask = np.array(mask, dtype=np.bool)
print(train_mask.shape)

mask1 = np.zeros(y3days.shape[0])
mask1[list(test_idx)] = 1
test_mask = np.array(mask1, dtype=np.bool)
print(test_mask.shape)

mask2 = np.zeros(y3days.shape[0])
mask2[list(val_idx)] = 1
val_mask = np.array(mask2, dtype=np.bool)
print(val_mask.shape)

# ### Make features or X

features3days = np.load('./data/final3days.npy')
features7days = np.load('./data/final7days.npy')
features15days = np.load('./data/final15days.npy')
features30days = np.load('./data/final30days.npy')

temp = np.zeros((277, 200), dtype = np.float32)
features3days = np.vstack((temp, features3days))
features7days = np.vstack((temp, features7days))
features15days = np.vstack((temp, features15days))
features30days = np.vstack((temp, features30days))

print(features3days.shape)
print(features7days.shape)
print(features15days.shape)
print(features30days.shape)

def complete_input(features):
    for index in stock_call_ind.keys():
        if len(stock_call_ind[index]) !=0:
            for i in range(len(stock_call_ind[index])):
                features[index] = np.add(features[index], features[stock_call_ind[index][i]])
            features[index]/=float(len(stock_call_ind[index]))
    return features

X3days = complete_input(features3days)
X7days = complete_input(features7days)
X15days = complete_input(features15days)
X30days = complete_input(features30days)
print(X3days.shape)
print(X7days.shape)
print(X15days.shape)
print(X30days.shape)

## edges -- edges_transpose
## y's -- y3days, y7days, y15days, y30days
## train, test, val masks -- train_mask, test_mask, val_mask
## data = X3days, X7days, X15days, X30days

edges_transpose = edges_transpose.astype('int32')
edge_index = torch.tensor(edges_transpose, dtype=torch.long)

## change for 7, 15, 30 days
x = torch.tensor(X3days, dtype=torch.float)
y = torch.tensor(y3days, dtype=torch.float)
data = Data(x=x, edge_index=edge_index.contiguous(), y=y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 200)
        self.conv2 = GCNConv(200, 100)
        self.hidden3 = nn.Linear(100,1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        gc1 = self.conv1(x, edge_index)
        gc2 = F.relu(gc1)
        gc3 = F.dropout(gc2, training=self.training)
        gc4 = self.conv2(gc3, edge_index)
        x = self.hidden3(gc4)

        return gc1, gc2, gc3, gc4, x

 mse_3days = []
# mse_7days = []
# mse_15days = []
#mse_30days = []
 for i in range(1000):
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    train_loss = []
    for epoch in range(100):
        optimizer.zero_grad()
        gc1, gc2, gc3, gc4, out = model(data)
        y_train =  data.y[data.train_mask].unsqueeze(1)
        loss = F.mse_loss(out[data.train_mask], y_train, reduction='mean')
        train_loss.append(loss)
        loss.backward()
        optimizer.step()
        model.eval()
    gc1, gc2, gc3, gc4, pred = model(data)
    y_test =  data.y[data.test_mask].unsqueeze(1)
    mse_loss = F.mse_loss(pred[data.test_mask], y_test, reduction='mean')
#    print('MSE loss for duration of 3 days: ', mse_loss)
    mse_3days.append(mse_loss.item())

 mse_3days = np.array(mse_3days)
 std = np.std(mse_3days)
 mean = np.mean(mse_3days)
 print('The value for the MSE for 3 days is: \nMean: ' + str(mean), ' Std: '+str(std))

print('GCN results:')
print()
std = np.std(mse_3days)
mean = np.mean(mse_3days)
print('The value for the MSE for 3 days is: \nMean: ' + str(mean), ' Std: '+str(std))
print()
#std1 = np.std(mse_7days)
#mean1 = np.mean(mse_7days)
#print('The value for the MSE for 7 days is: \nMean: ' + str(mean1), ' Std: '+str(std1))
#print()
#std2 = np.std(mse_15days)
#mean2 = np.mean(mse_15days)
#print('The value for the MSE for 15 days is: \nMean: ' + str(mean2), ' Std: '+str(std2))
#print()
#std3 = np.std(mse_30days)
#mean3 = np.mean(mse_30days)
#print('The value for the MSE for 30 days is: \nMean: ' + str(mean3), ' Std: '+str(std3))
#print()

print('The shape for first conv layer: ', gc1.detach().numpy().shape)
print('The shape for first conv layer with relu: ', gc2.detach().numpy().shape)
print('The shape for first conv layer with dropout: ', gc3.detach().numpy().shape)
print('The shape for second conv layer: ', gc4.detach().numpy().shape)


# ### Save embeddings

GC1 = {}
mat1 = gc2.detach().numpy()
for i in range(TOTAL_NODES):
    if i in indices_file_name.keys():
        GC1[indices_file_name[i]]= mat1[i,:]

with open('emb_3days_final.pkl', 'wb') as handle:
    pickle.dump(GC1, handle)
