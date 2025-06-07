import argparse
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


WORD_WRAP_LIMIT = 20
FONT_SIZE = 14

L = 'l'

# PC = 'psi_pc'
# RCD = 'rcd'
# RCD_ORACLE = 'rcd_oracle'
# ESSENTIAL = 'essential'
# KPC_ORACLE_0 = 'kpc_0_oracle'
# KPC_ORACLE_1 = 'kpc_1_oracle'
# KPC_ORACLE_2 = 'kpc_2_oracle'
# KPC_0 = 'kpc_0'
# KPC_1 = 'kpc_1'
# KPC_2 = 'kpc_2'
# ORIENT_0 = 'orient_0'
# ORIENT_1 = 'orient_1'
# ORIENT_2 = 'orient_2'
# MUTUAL_INFO = 'mutual_info'
# MI_GRAPH = 'mi_graph'
# MI_GRAPH_ORACLE = 'mi_graph_oracle'
# MARGINAL_CI = 'mci'
# ESSENTIAL_ORACLE = 'ikpc_-1'
# BOOSTED_ESSENTIAL_ORACLE = 'boosted_ikpc_-1'
# ESSENTIAL_ORACLE_POSSIBLE_PARENTS = 'ikpc_-1_parents'
# ESSENTIAL_ORACLE_NEW = 'ikpc_-1_new'
# ALPHA_CPDAG = 'alpha_-1'
# ALPHA_0 = 'alpha_0'
# ALPHA_1 = 'alpha_1'
# IKPC_0 = 'ikpc_0'
# IKPC_1 = 'ikpc_1'
# IKPC_2 = 'ikpc_2'
# CMI_DAG = 'cmi_dag'
# CMI_0 = 'cmi_0'
# CMI_1 = 'cmi_1'
# MI_CMI_0 = 'mi_cmi_0'
# MI_CMI_1 = 'mi_cmi_1'
# MI_CMI_DAG = 'mi_cmi_dag'
# IGS = 'igs'
PAGERANK = 'page_rank'
# RANDOM = 'random'
# TOCA = 'toca'
# BOSS = 'boss'
# BARO = 'baro'
SMOOTH_CH = 'smooth'
RCG_DAG = 'rcg_dag'

# KL_IKPC = 'kl_ikpc_2'
# MI_IKPC = 'kl_ikpc_2'

PREFIXES = {
    # 'TOCA': TOCA,
    # 'Random': RANDOM,
    # 'BOSS': BOSS,
    'RUN': PAGERANK,
    # # 'BARO': BARO,
    # # 'psi-PC': PC,
    # 'MI': MUTUAL_INFO,
    'SMOOTH': SMOOTH_CH,
    # 'RCD': RCD,
    # 'M-IGS': IGS,
    # 'RCD (oracle)': RCD_ORACLE,
    # 'Essential': ESSENTIAL,
    # 'kPC+MI': KPC_0,
    # '1-PC': KPC_1,
    # '2-PC': KPC_2,
    # 'kPC+MI (oracle)': KPC_ORACLE_0,
    # '1-PC (oracle)': KPC_ORACLE_1,
    # '2-PC (oracle)': KPC_ORACLE_2,
    # 'RCG-0': IKPC_0,
    # 'RCG-1': IKPC_1,
    # 'RCG-2': IKPC_2,
    # # 'O-RCG': ESSENTIAL_ORACLE,
    # 'T-RCG': ESSENTIAL_ORACLE_POSSIBLE_PARENTS,
    # # 'RCG++': ESSENTIAL_ORACLE_NEW,
    # 'RCG-0': ALPHA_0,
    # 'RCG-1': ALPHA_1,
    # 'RCG(CPDAG)': ALPHA_CPDAG,
    'RCG(DAG)': RCG_DAG,
}

DATA_CSV = 'data.csv'
TIME = 'time'
ACCURACY = 'accuracy'

GRAPH_LABELS = {
    TIME: 'Execution Time (ms)',
    ACCURACY: 'Accuracy@$l$',
}

# COLORS = ['C1', 'C0']
# For oracle version
COLORS = ['C3', 'C4', 'C1', 'C6', 'C0', 'C5', 'C2']
# For sample version
# COLORS = ['C4', 'C3', 'C1', 'C0', 'C2']
MARKERS = ['o', 's', '^', 'o', 'x', 'D', 'P']
LINE_STYLES = ['--', '-.', ':']

# For int samples
# COLORS = ['C1', 'C0', 'C2']
# MARKERS = ['^', 'x', 'P']
LINE_STYLES = [':', '--', ':']


# ============================= Private methods =============================

def _line_plot(data, labels, err=None, xlabel='', ylabel='',
               title='', xc_limit=None, log_scale=False, legend_position=None):
    
    markers = itertools.cycle(MARKERS)
    line_styles = itertools.cycle(LINE_STYLES)
    colors = itertools.cycle(COLORS)
    print(f'===================== {ylabel} =====================')
    print(f'mean = {data} | err = {err}')

    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 14

    legend_position = (0.98, 0.15)
    x = [str(x) for x in labels]

    for j, (l, v) in enumerate(data.items()):
        if err:
            l = ax.errorbar(x, v, yerr=err[l], label=l, marker=next(markers), ls=next(line_styles), color=next(colors))
            l[-1][0].set_linestyle('-.')
        else:
            ax.plot(x, v, label=l, marker=next(markers), ls=next(line_styles), color=next(colors))

        if log_scale:
            ax.set_yscale('log')

        if xc_limit is not None:
            ax.axvspan(str(xc_limit), str(np.max(labels)), alpha=0.1, color='darkorange')

    plt.grid(True, which="major", ls="-", color='0.9')

    ax.xaxis.label.set_fontsize(FONT_SIZE + 5)
    ax.yaxis.label.set_fontsize(FONT_SIZE + 5)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE + 2)

    ax.legend(loc='center right', bbox_to_anchor=legend_position,
                    borderaxespad=0, fancybox=True, shadow=True, ncol=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

def _k_plots(data, labels, err=None, xlabel='', ylabel='',
             title='', xc_limit=None, log_scale=False, legend_position=None):
    print(f'===================== {ylabel} =====================')
    print(f'mean = {data} | err = {err}')

    fig, axs = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    plt.subplots_adjust(right=1)

    plt.rcParams['font.size'] = FONT_SIZE

    x = [str(x) for x in labels]
    all_handles = []
    all_labels = []
    handles_labels_set = set()
    for i, (k, value) in enumerate(data.items()):
        markers = itertools.cycle(MARKERS)
        line_styles = itertools.cycle(LINE_STYLES)
        colors = itertools.cycle(COLORS)
        for j, (l, v) in enumerate(value.items()):
            if err:
                mm = next(markers)
                handle = axs[i].errorbar(x, v, yerr=err[k][l], label=l, marker=mm, ls=next(line_styles), color=next(colors))
            else:
                handle = axs[i].plot(x, v, label=l, marker=next(markers), ls=next(line_styles), color=next(colors))
            label = l
            if label not in handles_labels_set:
                if isinstance(handle, list):
                    all_handles.extend(handle)
                else:
                    all_handles.append(handle)
                all_labels.append(label)
                handles_labels_set.add(label)

            if log_scale:
                axs[i].set_yscale('log')

            if xc_limit is not None:
                axs[i].axvspan(str(xc_limit), str(np.max(labels)), alpha=0.1, color='darkorange')

        axs[i].xaxis.label.set_fontsize(FONT_SIZE + 5)
        axs[i].yaxis.label.set_fontsize(FONT_SIZE + 5)
        axs[i].tick_params(axis='both', which='major', labelsize=FONT_SIZE + 2)
        axs[i].set_title(f"Top-{k}")

    axs[1].set_ylim([0, 1])
    # axs[1].set_ylim([0, 1])

    # for ax in axs.flat:
    axs[1].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)

    fig.legend(all_handles, all_labels, loc='upper center', fancybox=True,
               ncol=7, bbox_to_anchor=(0.5, 1.01), fontsize=FONT_SIZE)

    # plt.legend(loc='upper left', bbox_to_anchor=(0, 1),
    #            borderaxespad=0, fancybox=True, shadow=True, ncol=1)

    fig.suptitle(title)
    fig.tight_layout()

def _top_k_plot(data, dir, save, attr, attr_label, xc_limit=None, metrics=None):
    _save_or_show = lambda name: plt.savefig(dir + name) if save else plt.show()
    m = metrics
    if m is None:
        m = {i: GRAPH_LABELS[i] for i in [TIME, ACCURACY]}

    temp = data.groupby([attr, L], as_index=False)
    mean = temp.mean(numeric_only=True)
    std_err = temp.sem(numeric_only=True)

    f = np.unique(mean[attr].values.tolist())
    ks = np.unique(mean[L].values.tolist())

    def _local_extract_field(field):
        data = dict()
        err = dict()
        for k in ks:
            filtered_data = mean[mean[L] == k]
            filtered_err = std_err[std_err[L] == k]

            data[k] = dict()
            err[k] = dict()
            for key, value in PREFIXES.items():
                label = f'{value}_{field}'
                if label not in filtered_data.columns: continue

                data[k][key] = filtered_data[label].values.tolist()
                err[k][key] = filtered_err[label].values.tolist()
        return data, err

    for i in m.keys():
        data, err = _local_extract_field(i)
        if i == TIME:
            _line_plot(data[1], f, err=err[1], xlabel=attr_label, ylabel=m[i], xc_limit=xc_limit, log_scale=True)
        else:
            _k_plots(data, f, err=err,
                    xlabel=attr_label, ylabel=m[i], xc_limit=xc_limit,
                    log_scale=False)
        _save_or_show(f"{i}.pdf")

def int_samples_plot(data, dir, save, y_attr):
    _save_or_show = lambda name: plt.savefig(dir + name) if save else plt.show()

    attr = 'int_samples'
    temp = data.groupby([attr], as_index=False)
    mean = temp.mean(numeric_only=True)
    std_err = temp.sem(numeric_only=True)

    f = np.unique(mean[attr].values.tolist())

    data = dict()
    err = dict()
    for key, value in PREFIXES.items():
        label = f'{value}_{y_attr}'
        if label not in mean.columns: continue

        data[key] = mean[label].values.tolist()
        err[key] = std_err[label].values.tolist()

    _line_plot(data, f, err=err,
               xlabel='Samples', ylabel=GRAPH_LABELS[y_attr],
               xc_limit=xc_limit, log_scale=(y_attr == 'time'))
    _save_or_show(f"{attr}_{y_attr}.pdf")

# ============================= Public methods =============================

def top_k_multiple_nodes(data, dir, **kwargs):
    save = kwargs['save']
    print(f'{save=}')
    xc_limit = kwargs['xc_limit']
    _top_k_plot(data, dir, save, 'nodes', 'Nodes', xc_limit)

def multiple_int_samples(data, dir, **kwargs):
    save = kwargs['save']
    int_samples_plot(data, dir, save, 'accuracy')
    int_samples_plot(data, dir, save, 'time')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates plots from experiment data')

    parser.add_argument('--exp', type=int, required=True,
                        help='The type plots to generate')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the experiment data')
    parser.add_argument('--show', action='store_true',
                        help='Show the plots, otherwise save them')
    parser.add_argument('--xc-limit', type=int, default=None,
                        help='A value from x-axis where the colored region will start')

    args = parser.parse_args()
    exp = args.exp
    path = args.path
    save = not args.show
    xc_limit = args.xc_limit
    dir = path + '/'
    data = pd.read_csv(dir + DATA_CSV)

    fn = {1: top_k_multiple_nodes,
          2: multiple_int_samples}

    fn[exp](data, dir, save=save, xc_limit=xc_limit)
