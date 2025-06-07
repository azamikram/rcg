import os
import time
import argparse
from multiprocessing import Pool

import pandas as pd

# from rcd import two_phase as rcd
# import find_root_cause as ft
# import mutual_info as mt
# import marginal_ci as mci
# import ikpc
# import cmi
# import mi_cmi
# import mi_graph
# import igs
import page_rank
# import toca
# import random_selection
# import boss
import rcg
# import baro
import smooth

from utils import base_utils as bu
from config import ExperimentConf, load_config, dump_config

RCG_DAG = 'rcg_dag'
PAGERANK = 'page_rank'
SMOOTH_CH = 'smooth'

BASELINES = [
    PAGERANK,
    SMOOTH_CH,
    # RCG_DAG,
]

RESULT_DIR = 'exp_results'
DEFAULT_CONFIG = 'experiments.yaml'


def run_baselines(src_dir, seed, cfg: ExperimentConf):
    a_node = bu.load_data(f'{src_dir}/{bu.GRAPH_GEN_INFO}')[bu.ANOMALOUS_NODE]
    result = {'l': cfg.l_value, 'seed': seed,
              'a_node': a_node, 'int_samples': cfg.interventional_samples}
    def _extract_result(result, prefix):
        l_rc = result['root_cause'][:cfg.l_value]
        accuracy = 1 if a_node in l_rc else 0
        return {
            f"{prefix}_tests": result['tests'],
            f"{prefix}_time": result['time'],
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_top_l_targets": l_rc,
        }
    (normal_df, anomalous_df) = bu.load_datasets(src_dir)

    # We use 10,000 samples for normal period and 1,000 samples for anomalous
    # By default, data_generator generates 10,000 samples for both normal
    # and anomalous dataset.
    anomalous_df = anomalous_df.sample(n=cfg.interventional_samples, random_state=seed, replace=False)
    anomalous_df.reset_index(drop=True, inplace=True)

    # rcd_r = rcd.top_k_rc(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir,
    #                      cfg.l_value, None, seed=seed, oracle=False, localized=True, verbose=cfg.verbose)
    # result = {**result, **_extract_result(rcd_r, f'rcd')}

    # igs_r = igs.run_algo(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir,
    #                      perfect_ci=False, max_l=cfg.l_value)
    # result = {**result, **_extract_result(igs_r, 'igs')}

    # _oracle = '_oracle' if cfg.oracle else ''
    # for i in range(0, 1):
    #     _kpc_r = ft.run_algo(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir,
    #                          cfg.l_value, i, seed=seed, oracle=cfg.oracle, perfect_ci=False, verbose=cfg.verbose)
    #     result = {**result, **_extract_result(_kpc_r, f'kpc_{i}{_oracle}')}

    # mutual_info_r = mt.rank_variables(normal_df.copy(deep=True), anomalous_df.copy(deep=True))
    # result = {**result, **_extract_result(mutual_info_r, 'mutual_info')}

    if PAGERANK in BASELINES:
        page_rank_r = page_rank.rank_variables(src_dir)
        result = {**result, **_extract_result(page_rank_r, PAGERANK)}

    # for i in [0, 1]:
    #     _ikpc_r = ikpc.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
    #                        src_dir, k=i, oracle=cfg.oracle)
    #     result = {**result, **_extract_result(_ikpc_r, f'ikpc_{i}')}

        # _ikpc_r = ikpc.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
        #                    src_dir, k=i, oracle=cfg.oracle, boosted=True)
        # result = {**result, **_extract_result(_ikpc_r, f'boosted_ikpc_{i}')}

        # _ikpc_r = ikpc.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
        #             src_dir, k=i, oracle=cfg.oracle, parents=True)
        # result = {**result, **_extract_result(_ikpc_r, f'ikpc_{i}_parents')}

        # _ikpc_r = ikpc.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
        #             src_dir, k=i, oracle=cfg.oracle, new_rank=True)
        # result = {**result, **_extract_result(_ikpc_r, f'ikpc_{i}_new')}

    # for i in range(0, 2):
    #     cmi_graph_r = cmi.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir, k=i)
    #     result = {**result, **_extract_result(cmi_graph_r, f'cmi_{i}')}

    #     cmi_graph_r = mi_cmi.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir, k=i)
    #     result = {**result, **_extract_result(cmi_graph_r, f'mi_cmi_{i}')}

    # cmi_graph_r = cmi.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir, k=None)
    # result = {**result, **_extract_result(cmi_graph_r, f'cmi_dag')}

    # mi_cmi_graph_r = mi_cmi.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir, k=None)
    # result = {**result, **_extract_result(mi_cmi_graph_r, f'mi_cmi_dag')}

    # mi_graph_r = mi_graph.rank_variables(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
    #                                      cfg.l_value, src_dir, oracle=cfg.oracle)
    # result = {**result, **_extract_result(mi_graph_r, f'mi_graph{_oracle}')}

    # mci_r = mci.rank_variables(normal_df.copy(deep=True), anomalous_df.copy(deep=True))
    # result = {**result, **_extract_result(mci_r, 'mci')}

    # smooth_traverse_r = toca.rank_variables(normal_df.copy(deep=True),
    #                                         anomalous_df.copy(deep=True),
    #                                         path=src_dir)
    # result = {**result, **_extract_result(smooth_traverse_r, 'smooth_full')}

    if SMOOTH_CH in BASELINES:
        smooth_new_r = smooth.rank_variables(normal_df.copy(deep=True),
                                                anomalous_df.copy(deep=True),
                                                src_dir)
        result = {**result, **_extract_result(smooth_new_r, SMOOTH_CH)}

    # toca_r = toca.rank_variables(normal_df.copy(deep=True), anomalous_df.copy(deep=True), path=src_dir)
    # result = {**result, **_extract_result(toca_r, 'toca')}

    # random_r = random_selection.rank_variables(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir)
    # result = {**result, **_extract_result(random_r, 'random')}

    # boss_r = boss.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir)
    # result = {**result, **_extract_result(boss_r, 'boss')}

    # for i in [-1]:
    #     alpha_r = cpdag_rca.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
    #                             src_dir, cfg.l_value, k=i, oracle=cfg.oracle)
    #     result = {**result, **_extract_result(alpha_r, f'alpha_{i}')}

    if RCG_DAG in BASELINES:
        rcg_dag_r = rcg.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
                            src_dir, cfg.l_value, dag=True)
        result = {**result, **_extract_result(rcg_dag_r, RCG_DAG)}

    # baro_r = baro.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True))
    # result = {**result, **_extract_result(baro_r, 'baro')}

    if cfg.verbose:
        print(f"Output: {result}")
    return result

# Change the total number of nodes
def different_nodes(src_dir, cfg: ExperimentConf):
    out_dir = f"{src_dir}/{RESULT_DIR}/{bu.readable_time()}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dump_config(cfg, out_dir)
    if cfg.threading:
        t_pool = Pool(cfg.workers)

    df_counter = 0
    df = pd.DataFrame()
    def _store(row):
        nonlocal df
        nonlocal df_counter
        if row is not None:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        if df_counter % 10 == 0: # Write in batch of 10
            df.to_csv(f'{out_dir}/data.csv', mode='a', header=(df_counter == 0), index=False)
            df = pd.DataFrame()
        df_counter += 1

    # Number of interventional samples to use
    cfg.interventional_samples = cfg.anomalous_samples
    for node, n_path in bu.dir_iterator(src_dir):
        print(f"Running the experiment with {node} nodes")
        for l in cfg.L:
            cfg.l_value = l
            if cfg.threading:
                future = list()
            for i, i_path in bu.dir_iterator(n_path):
                if cfg.threading:
                    future.append(t_pool.starmap_async(run_baselines, [(i_path, int(i), cfg)]))
                else:
                    _store({'nodes': node, **run_baselines(i_path, int(i), cfg)})

            if cfg.threading:
                for f in future:
                    _store({'nodes': node, **(f.get()[0])})
        
    _store(None)

    if cfg.threading:
        t_pool.close()
        t_pool.join()
    return out_dir

# Change the number of interventional samples
def different_int_samples(src_dir, cfg: ExperimentConf):
    out_dir = f"{src_dir}/{RESULT_DIR}/{bu.readable_time()}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dump_config(cfg, out_dir)
    if cfg.threading:
        t_pool = Pool(cfg.workers)

    df_counter = 0
    df = pd.DataFrame()
    def _store(row):
        nonlocal df
        nonlocal df_counter
        if row is not None:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        if df_counter % 10 == 0: # Write in batch of 10
            df.to_csv(f'{out_dir}/data.csv', mode='a', header=(df_counter == 0), index=False)
            df = pd.DataFrame()
        df_counter += 1

    node = '25'
    n_path = os.path.join(src_dir, node)
    for _int_sample in cfg.int_samples:
        print(f"Running the experiment with {_int_sample} interventional samples")
        cfg.l_value = 1 # top-1
        cfg.interventional_samples = _int_sample
        if cfg.threading:
            future = list()
        for i, i_path in bu.dir_iterator(n_path):
            if cfg.threading:
                future.append(t_pool.starmap_async(run_baselines, [(i_path, int(i), cfg)]))
            else:
                _store({'nodes': node, **run_baselines(i_path, int(i), cfg)})

        if cfg.threading:
            for f in future:
                _store({'nodes': node, **(f.get()[0])})
    _store(None)

    if cfg.threading:
        t_pool.close()
        t_pool.join()
    return out_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all the baselines on the given dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the data')
    parser.add_argument('--exp', type=int, required=True, help='The type of experiment')
    args = parser.parse_args()
    path = args.path
    exp = args.exp
    cfg: ExperimentConf = load_config(DEFAULT_CONFIG, ExperimentConf)

    fn = {
            1: different_nodes, # Figure 3(a)
            2: different_int_samples # Figure3 (b)
        }
    start = time.perf_counter()
    src_dir = fn[exp](path, cfg)
    end = time.perf_counter()
    print(f"The experiment took {round(end - start, 3)} seconds")
    print(f"The result of the experiment is stored at {src_dir}")
