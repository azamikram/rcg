#!/usr/bin/env python3

import time
import argparse

import networkx as nx

from causallearn.graph.Dag import Dag
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

from utils import base_utils as bu

VERBOSE = True
if not VERBOSE:
    import warnings
    warnings.filterwarnings('ignore')

CORES = 1
K = 0
BINS = 5
PRE_PROCESS = False
ORACLE = False
CI_TEST = 'chisq'
ALPHA = 0.05

current_time = lambda: time.perf_counter() * 1e3


def learn(path):
    nx_graph: nx.DiGraph = bu.load_graph(f'{path}/{bu.GROUND_TRUTH_NX_GRAPH}')
    node_map = {x: GraphNode(x) for x in nx_graph.nodes}
    dag: Dag = Dag(list(node_map.values()))
    for u, v in nx_graph.edges():
        dag.add_edge(Edge(node_map[u], node_map[v], Endpoint.TAIL, Endpoint.ARROW))
    G = dag2cpdag(dag)

    cg_path = f'{path}/{bu.get_prior_graph_name(-1, True)}'
    bu.store_causal_learn_graph(G, -1, cg_path)
    return G

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates Oracle CPDAG from a dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the experiment data')
    args = parser.parse_args()
    path = args.path

    s_time = current_time()
    learn(path)
    e_time = current_time()
    print(f'Learning the essential graph took {e_time - s_time}')
