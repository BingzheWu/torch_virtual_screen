import yaml
from core.train_utils import init_featurizer, mkdir_p
import torch

def load_yaml_cfg(args_file):
    with open(args_file, 'r') as f:
        args = yaml.load(f)
    return args
def load_args(args_file):
    args = load_yaml_cfg(args_file)
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')
    mkdir_p(args['result_path'])
    args = init_featurizer(args)
    args['in_node_feats'] = args['node_featurizer'].feat_size()
    if args['edge_featurizer'] is not None:
        args['in_edge_feats'] = args['edge_featurizer'].feat_size()
    if 'task_names' in args:
        args['task_names'] = args['task_names'].split(',')
        args['n_tasks'] = 1
    else:
        args['task_names'] = None
        args['n_tasks'] = 12
    return args