from __future__ import annotations

from itertools import combinations
from typing import List, Dict, Tuple, Set

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Node import Node
from causallearn.utils.PCUtils.Helper import append_value
# from causallearn.utils.cit import *
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
import joblib

from .cit import CIT_Base, D_Separation

def fas(data: ndarray, nodes: List[Node], independence_test_method: CIT_Base, alpha: float = 0.05,
        knowledge: BackgroundKnowledge | None = None, depth: int = -1, parallel = False,  p_cores=1, s=None, batch=None,
        verbose: bool = False, stable: bool = True, show_progress: bool = True) -> Tuple[
    GeneralGraph, Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int, Set[int]], float]]:
    """
    Implements the "fast adjacency search" used in several causal algorithm in this file. In the fast adjacency
    search, at a given stage of the search, an edge X*-*Y is removed from the graph if X _||_ Y | S, where S is a subset
    of size d either of adj(X) or of adj(Y), where d is the depth of the search. The fast adjacency search performs this
    procedure for each pair of adjacent edges in the graph and for each depth d = 0, 1, 2, ..., d1, where d1 is either
    the maximum depth or else the first such depth at which no edges can be removed. The interpretation of this adjacency
    search is different for different algorithm, depending on the assumptions of the algorithm. A mapping from {x, y} to
    S({x, y}) is returned for edges x *-* y that have been removed.

    Parameters
    ----------
    data: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    nodes: The search nodes.
    independence_test_method: the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    knowledge: background background_knowledge
    depth: the depth for the fast adjacency search, or -1 if unlimited
    p_cores : int
        Number of CPU cores to be used
    s : bool, default False
        memory-efficient indicator
    batch : int
        number of edges per batch
    verbose: True is verbose output should be printed or logged
    stable: run stabilized skeleton discovery if True (default = True)
    show_progress: whether to use tqdm to show progress bar
    Returns
    -------
    graph: Causal graph skeleton, where graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j.
    sep_sets: Separated sets of graph
    test_results: Results of conditional independence tests
    """

    dsep = type(independence_test_method) == D_Separation
    if dsep:
        dag = independence_test_method.true_dag
        idx_to_node = independence_test_method.i_to_node
        node_to_idx = {node:idx for idx, node in idx_to_node.items()}

    def test(x, y):
        nonlocal dag, idx_to_node, node_to_idx

        K_x_y = 1
        sub_z = None
        # On X's neighbours
        # adj_x = set(np.argwhere(skeleton[x] == 1).reshape(-1, ))
        Neigh_x = cg.neighbors(x)
        if dsep:
            Neigh_x = np.array([node_to_idx[x] for x in dag.predecessors(idx_to_node[x])])

        Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
        if len(Neigh_x_noy) >= current_depth:
            for sub_z in combinations(Neigh_x_noy, current_depth):
                p = cg.ci_test(x, y, sub_z)
                
                if p > alpha:
                    K_x_y = 0
                    break
            if K_x_y == 0:
                return K_x_y, sub_z
        return K_x_y, sub_z


    def parallel_cell(x, y):

        # On X's neighbours
        K_x_y, sub_z = test(x, y)
        if K_x_y == 1:
            # On Y's neighbours
            K_x_y, sub_z = test(y, x)

        return (x, y), K_x_y, sub_z


    ## ------- check parameters ------------
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    if not all(isinstance(node, Node) for node in nodes):
        raise TypeError("'nodes' must be 'List[Node]' type!")
    if not isinstance(independence_test_method, CIT_Base):
        raise TypeError("'independence_test_method' must be 'CIT_Base' type!")
    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")
    if knowledge is not None and type(knowledge) != BackgroundKnowledge:
        raise TypeError("'knowledge' must be 'BackgroundKnowledge' type!")
    if type(depth) != int or depth < -1:
        raise TypeError("'depth' must be 'int' type >= -1!")
    ## ------- end check parameters ------------

    if depth == -1:
        depth = float('inf')

    no_of_var = len(nodes)
    ls_of_var = [i for i in range(no_of_var)]
    node_names = [node.get_name() for node in nodes]
    cg = CausalGraph(no_of_var, node_names=node_names)
    cg.set_ind_test(independence_test_method)
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}
    test_results: Dict[Tuple[int, int, Set[int]], float] = {}

    def remove_if_exists(x: int, y: int) -> None:
        edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge is not None:
            cg.G.remove_edge(edge)

    if not parallel:
        var_range = tqdm(range(no_of_var), leave=True) if show_progress \
            else range(no_of_var)
    current_depth: int = -1

    # first, we incorporate any prior knowledge before we may initiate parallelism
    if knowledge is not None:
        for i, j in combinations(ls_of_var, 2):
            if (knowledge.is_forbidden(cg.G.nodes[i], cg.G.nodes[j])
                    and knowledge.is_forbidden(cg.G.nodes[j], cg.G.nodes[i])):
                remove_if_exists(i, j)
                remove_if_exists(j, i)
                append_value(cg.sepset, i, j, ())
                append_value(cg.sepset, j, i, ())
                sep_sets[(i, j)] = set()
                sep_sets[(j, i)] = set()

    # Main Loop
    while cg.max_degree() - 1 > current_depth and current_depth < depth:
        current_depth += 1
        if parallel:
            J = [(x,y) for x, y in combinations(ls_of_var, 2) if cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]) is not None]
            if not s or not batch:
                batch = len(J)
            if batch < 1:
                batch = 1
            if not p_cores or p_cores == 0:
                raise ValueError(f'If variant is parallel, type of p_cores '
                                 f'must be int, but got {type(p_cores)}.')
            for i in range(int(np.ceil(len(J) / batch))):
                each_batch = J[batch * i: batch * (i + 1)]
                parallel_result = joblib.Parallel(n_jobs=p_cores,
                                                  max_nbytes=None)(
                    joblib.delayed(parallel_cell)(x, y) for x, y in
                    each_batch
                )

                for (x,y), K_x_y, sub_z in parallel_result:
                    if K_x_y == 0:
                        remove_if_exists(x, y)
                        remove_if_exists(y, x)
                        append_value(cg.sepset, x, y, sub_z)
                        append_value(cg.sepset, y, x, sub_z)
                        sep_sets[(x, y)] = sub_z
                        sep_sets[(y, x)] = sub_z

        else:
            edge_removal = set()
            for x in var_range:
                if show_progress:
                    var_range.set_description(f'Depth={current_depth}, working on node {x}')
                    var_range.update()
                Neigh_x = cg.neighbors(x)
                if dsep:
                    Neigh_x = np.array([node_to_idx[x] for x in dag.predecessors(idx_to_node[x])])
                if len(Neigh_x) < current_depth - 1:
                    continue
                for y in Neigh_x:
                    sepsets = set()
                    Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                    for S in combinations(Neigh_x_noy, current_depth):
                        p = cg.ci_test(x, y, S)
                        test_results[(x, y, S)] = p
                        if p > alpha:
                            if verbose:
                                print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                            if not stable:
                                remove_if_exists(x, y)
                                remove_if_exists(y, x)
                                append_value(cg.sepset, x, y, S)
                                append_value(cg.sepset, y, x, S)
                                sep_sets[(x, y)] = set(S)
                                sep_sets[(y, x)] = set(S)
                                break
                            else:
                                edge_removal.add((x, y))  # after all conditioning sets at
                                edge_removal.add((y, x))  # depth l have been considered
                                for s in S:
                                    sepsets.add(s)
                        else:
                            if verbose:
                                print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

            for (x, y) in edge_removal:
                remove_if_exists(x, y)
                if cg.sepset[x, y] is not None:
                    origin_set = set(l_in for l_out in cg.sepset[x, y]
                                    for l_in in l_out)
                    sep_sets[(x, y)] = origin_set
                    sep_sets[(y, x)] = origin_set

    return cg.G, sep_sets, test_results
