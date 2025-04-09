import json
import yaml
from dataclasses import dataclass

@dataclass
class GraphGenConf:
    seed: int
    nodes: int
    states: int
    samples: int

    verbose: bool

def load_config(name: str, type: type):
    with open(f'config/{name}', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return type(**cfg)

def dump_config(conf, src_dir: str):
    with open(f'{src_dir}/readme.txt', 'w') as readme:
        readme.write(json.dumps(_obj_to_dict(conf)))
    readme.close()

def _obj_to_dict(obj):
    if type(obj) is dict:
        res = {}
        for k, v in obj.items():
            res[k] = _obj_to_dict(v)
        return res
    elif not hasattr(obj, '__dict__'):
        return obj
    else:
        return _obj_to_dict(vars(obj))
