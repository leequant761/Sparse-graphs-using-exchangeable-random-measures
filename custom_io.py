import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import loadmat

from simulation import GGPgraphrnd

def load_graph(f_name:str):
    if f_name=='email':
        df = pd.read_csv('./data/Email-Enron.txt', sep='\t', skiprows=3)
        row = df.values[:, 0]
        col = df.values[:, 1]
        num_nodes = 1 + max(row.max(), col.max())
        data = np.ones(len(df), dtype=int)
        graph = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        directed = False

    elif f_name=='airport':
        df = pd.read_csv('./data/USairport_2010.txt', sep=' ', header=None, dtype=int)
        row = df.values[:, 0]
        col = df.values[:, 1]
        hash_table = {key: value for value, key in enumerate(np.unique(np.append(row, col)))}
        adjust_ftn = np.vectorize(lambda key: hash_table[key])
        row, col = adjust_ftn(row), adjust_ftn(col)
        data = df.values[:, 2]
        num_nodes = 1 + max(row.max(), col.max())
        data = np.ones(len(df), dtype=int)
        graph = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        directed = True

    elif f_name=='powergrid':
        graph = coo_matrix(loadmat('./data/USpowerGrid.mat')['Problem'][0][0][2], dtype=int)
        directed = False

    elif f_name=='simul':
        graph, _, _ = GGPgraphrnd(alpha=300, sigma=0.5, tau=1)
        directed = False
    else:
        raise ValueError(f"{f_name} is not in ('email', 'airport', 'powergrid', 'simul')")

    return graph, directed