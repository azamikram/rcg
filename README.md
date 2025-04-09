Follow the following steps to run M-IGS and RCG(CPDAG)

1. Generate data: Use `graph_gen.py` to generate syntheitc data for an incident. To change any tuneable paramter, please specify the changes in `config/graph_gen.yaml`.
2. Learn a Prior: For RCG(CPDAG), we first need to learn the true CPDAG. Use `learn_prior.py` to learn ground truth CPDAG. The script takes `--path` as the argument.
3. RCG(CPDAG): Run `rcg.py` with `--path` to see the result of RCG(CPDAG).
4. M-IGS: Run `m_igs.py` with --path to see the result of M-IGS. The script takes an additional parameter `--oracle` to specify whether the algorithm should use d-separation CI test (ground truth) or the data.
