import numpy as np
import json
import pandas as pd
from wikidata.client import Client
import networkx as nx

client = Client()
entities = []
df = pd.read_csv('earnings_wiki.csv',header=None)
df.columns = ['qval']

print('unique stocks',len(df['qval'].unique()))

qvals = []
for i,j in df.iteritems():
    qvals.append(j.values)

qvals = qvals[0].tolist()

for q in qvals:
    e = client.get(q, load=True)
    entities.append(e)
	
connections = []
pvals = ['P127','P155','P156','P355','P749']   # first order relations
def first_order(entities,qvals,pvals):  
    for e in entities:
        for p in pvals:
		    try:
                if p in e.attributes['claims'].keys():
			        for i in range(0,len(e.attributes['claims'][p])):
                        q2 = e.attributes['claims'][p][i]['mainsnak']['datavalue']['value']['id']
                        if q2 in qvals:
                            connections.append((e.attributes['title'],q2,p))
			except:
			    print('No keys present for this stock')
	
    
	return
	
first_order(entities,qvals,pvals)
print('first order relations extracted')	

p1_val =  ['P31','P31','P31','P112','P112','P112','P113','P114','P121','P121','P127','P127','P127','P127','P127','P127','P127','P155','P155','P166','P169','P169','P169','P169','P199','P06','P355','P355','P355','P355','P361','P366','P400','P452','P452','P452','P452','P463','P749','P749','P1056','P1056','P1056','P1056','P1056','P1056','P1344','P1830','P1830','P2770','P3320','P3320']	
p2_val = ['P366','P452','P1056','P112','P127','P169','P113','P114','P1056','P121','P112','P127','P169','P355','P749','P1830','P3320','P155','P355','P166','P112','P127','P169','P3320','P55','P1056','P127','P155','P199','P355','P361','P31','P1056','P31','P452','P1056','P2770','P463','P127','P1830','P31','P121','P306','P400','P452','P1056','P1344','P127','P749','P452','P127','P169']

def sec_order(entities,p1_val,p2_val):  # p1_val and p2_val are relation 1 and 2 respectively in second order relations
    for e1 in entities:
        for e2 in entities:
            if e1==e2:
                continue
            for i in range(0,len(p1_val)):
			    try:
                    if p1_val[i] in e1.attributes['claims'].keys() and p2_val[i] in e2.attributes['claims'].keys():
                    
                        int_a = []
                        int_b = []
                        for a in range(0,len(e1.attributes['claims'][p1_val[i]])):
                            int_a.append(e1.attributes['claims'][p1_val[i]][a]['mainsnak']['datavalue']['value']['id'])
                        for b in range(0,len(e2.attributes['claims'][p2_val[i]])):
                            int_b.append(e2.attributes['claims'][p2_val[i]][b]['mainsnak']['datavalue']['value']['id'])
                        a_set = set(int_a)
                        b_set = set(int_b)
                        if len(a_set.intersection(b_set))>0:
                            rel = [p1_val[i],p2_val[i]]
                            connections.append((e1.attributes['title'],e2.attributes['title'],'_'.join(rel)))
							
				except:
				    print('No keys present for this stock')
					
	return

sec_order(entities,p1_val,p2_val)
print('second order relations extracted')
df = pd.DataFrame(connections,columns=['source','target','relation'],index = None)

print('length of unique relations out of 57 valid relations',len(df['relation'].unique()))

tickers = np.genfromtxt('earnings_wiki.csv', dtype=str, delimiter=',',skip_header=False)
print('length of tickers',len(tickers))

wikiid_ticind_dic = {}
# this is just mapping tickers to ids
for ind, tw in enumerate(tickers):
    if not tw == 'unknown':
        wikiid_ticind_dic[tw] = ind

print('#tickers aligned:', len(wikiid_ticind_dic))

occur_paths = set()
for ind in data_frame.index:
    occur_paths.add(data_frame['relation'][ind])
		
valid_path_index = {}
for ind, path in enumerate(occur_paths):
    valid_path_index[path] = ind
	
for path, ind in valid_path_index.items():
    print(path, ind)
	
wiki_relation_embedding = np.zeros([tickers.shape[0], tickers.shape[0], len(valid_path_index) + 1],dtype=int)
conn_count = 0

for ind in data_frame.index:
    if data_frame['relation'][ind] in valid_path_index.keys():
        wiki_relation_embedding[wikiid_ticind_dic[data_frame['source'][ind]]][wikiid_ticind_dic[data_frame['target'][ind]]][valid_path_index[data_frame['relation'][ind]]] = 1
        conn_count += 1
		
		
for i in range(tickers.shape[0]):
    wiki_relation_embedding[i][i][-1] = 1
	
print('print embedding shape',wiki_relation_embedding.shape)

np.save('earnings_call_wiki_relation', wiki_relation_embedding)

def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),np.sum(relation_encoding, axis=2))
    adjacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),np.ones(rel_shape, dtype=float))
    print('adj matrix',adjacent)
    np.save('Earnings_wiki_adj_matrix.npy',adjacent)
	
load_graph_relation_data('earnings_call_wiki_relation.npy', lap=False)

print('Graph Stats')
total = np.load('Earnings_wiki_adj_matrix.npy')
G = nx.from_numpy_matrix(total)
print (nx.info(G))
