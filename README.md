Follow the following steps to run M-IGS and RCG(CPDAG)

1. Generate data: Use `graph_gen.py` to generate synthetic data for an incident. To change any tuneable parameter, please specify the changes in `config/graph_gen.yaml`.
2. Learn a Prior: For RCG(CPDAG), we first need to learn the true CPDAG. Use `learn_prior.py` to learn ground truth CPDAG. The script takes `--path` as the argument.
3. RCG(CPDAG): Run `rcg.py` with `--path` to see the result of RCG(CPDAG).
4. M-IGS: Run `m_igs.py` with --path to see the result of M-IGS. The script takes an additional parameter `--oracle` to specify whether the algorithm should use d-separation CI test (ground truth) or the data.

### Files and Folders
- `graph_gen.py` generates data for a single incident. It uses `config/graph_gen.yaml` as the input.
- `data_generator.py` uses `graph_gen.py` to generate a dataset which consists of a number of incidents.
 It uses `config/data_gen.yaml` as the input.
- Similarly, `learn_kess_g.py` takes the path to an incident as an input and learns the `k`-essential graph for that. It uses `k` as the input parameter for `k`-PC. Use `k=-1` for learning the CPDAG. `ORACLE` is another key variable which decides whether to use d-separation to learn the graph or the data. All of these variables are defined in `learn_kess_g.py`. Wen also provide our implementation of `k`-PC in `para_kpc` which implements `k`-PC that uses multiple cores to speed up the learning.

### Additional Notes
- We typically use chi-square `chisq` as the main CI test which works for discrete data. For our synthetic data experiments, we generate discrete data (`states` in `config/graph_gen.yaml`). But for Sock-shop or real-world dataset, we have to discretize the data. For that we use the variables `BINS`.
- For application specific data, sometimes we also need to remove a few variables such as the timestamp or variables with constant values. Fro that we generally use the boolean variable named `PRE_PROCESS`.
