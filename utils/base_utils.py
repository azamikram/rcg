import os
import time
import pickle
import datetime
import warnings

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from causallearn.graph.Edge import Edge
from causallearn.utils.cit import chisq
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph


IND_TEST = chisq
BINS = 5
SOCKSHOP_LATENCY_PREC = 90

ANOMALOUS_NODE = 'a_node'
GRAPH_GEN_INFO = 'gen_graph_info.pkl'
SOCKSHOP_DATA_INFO = 'sockshop_info.pkl'
NORMAL_DATA = 'normal.csv'
ANOMALOUS_DATA = 'anomalous.csv'
GROUND_TRUTH_NX_GRAPH = 'g_bn_graph.pkl'
GROUND_TRUTH_PDF = 'ground-truth.pdf'
ESSENTIAL_G = 'essential_g.pkl'
BOSS_ESSENTIAL_G = 'boss_essential_g.pkl'
K_ESSENTIAL_G = 'k_essential_g.pkl'
MODIFIED_CG = 'modified-cg.pkl'
ESSENTIAL_G_PDF = 'essential_g.pdf'
BOSS_ESSENTIAL_G_PDF = 'boss_essential_g.pdf'
K_ESSENTIAL_G_PDF = 'k_essential_g.pdf'
MODIFIED_CG_PDF = 'modified-cg.pdf'
NORMAL_BN = 'normal.bif'
ANOMALOUS_BN = 'anomalous.bif'

F_NODE = 'F-node'

VERBOSE = False

current_time = lambda: time.perf_counter()


def readable_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def get_node_name(node):
    return f'X{node}'

def get_prior_graph_name(k, oracle):
    if oracle and k == -1: return 'oracle-cpdag'

    return f"{'oracle' if oracle else 'sample'}-{k}"

def load_datasets(path, verbose=VERBOSE):
    if verbose:
        print(f'Loading the dataset from {path}...')
    normal_df = pd.read_csv(f'{path}/{NORMAL_DATA}')
    anomalous_df = pd.read_csv(f'{path}/{ANOMALOUS_DATA}')
    return (normal_df, anomalous_df)

def load_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph
load_data = load_graph

# Iterate over all the files in the given directory
# and output directories that have integer as their name
def dir_iterator(path):
    sorted_files = list()
    for f in os.listdir(path):
        try:
            sorted_files.append(int(f))
        except:
            pass
    sorted_files.sort()
    for f in sorted_files:
        p_path = f'{path}/{f}'
        if not os.path.isdir(p_path): continue
        yield(f, p_path)

def add_fnode(normal_df, anomalous_df):
    normal_df[F_NODE] = 0
    anomalous_df[F_NODE] = 1
    return pd.concat([normal_df, anomalous_df])

def create_causallearn_graph(nodes, edges, arcs) -> GeneralGraph:
    _nodes = [GraphNode(x) for x in nodes]
    G = GeneralGraph(_nodes)
    def _add_edges(edges, dir=False):
        for u, v in edges:
            _u = G.get_node(u)
            _v = G.get_node(v)
            if dir:
                G.add_edge(Edge(_u, _v, Endpoint.TAIL, Endpoint.ARROW))
            else:
                G.add_edge(Edge(_u, _v, Endpoint.TAIL, Endpoint.TAIL))
    _add_edges(edges)
    _add_edges(arcs, dir=True)
    return G

def store_causal_learn_graph(G: GeneralGraph, k: int, path: str):
    with open(path, 'wb') as f:
        pickle.dump({'graph': G, 'k':k}, f)

def discretize(data, bins=BINS):
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        discretizer.fit(data)
    disc_d = discretizer.transform(data)
    disc_d = pd.DataFrame(disc_d, columns=data.columns)
    disc_d = disc_d.astype(int)
    return disc_d

def _select_cols(df, keep_cols=[]):
    names = ['front-end', 'user', 'catalogue', 'orders', 'carts', 'payment', 'shipping']
    metrics = ['cpu', 'mem', 'lod', 'lat_50']

    l_names = [] + keep_cols
    for i in names:
        for j in metrics:
            _name = f'{i}_{j}'
            if _name in df.columns:
                l_names.append(_name)
    df = df[l_names]

    # Drop constants
    df = df.loc[:, (df != df.iloc[0]).any()]
    return df

def preprocess_sockshop(n_df, a_df, bins=BINS):
    df = add_fnode(n_df, a_df)
    df = _select_cols(df, keep_cols=[F_NODE])
    df = discretize(df, bins)
    n_df = df[df[F_NODE] == 0].drop(columns=[F_NODE])
    a_df = df[df[F_NODE] == 1].drop(columns=[F_NODE])
    return n_df, a_df

def sort_by_mi(df):
    scores = {}
    for c in df.columns[:-1]:
        scores[c] = mt.mutual_information(df, c, F_NODE)
    return sorted(scores, key=lambda t: t[1], reverse=True)
