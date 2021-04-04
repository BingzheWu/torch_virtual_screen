import numpy as np
import sys
sys.path.append('./')
from core.eval_utils import load_yaml_cfg, eval_and_save
from core.train_utils import init_featurizer
import torch

if __name__ == '__main__':
    import sys
    args_file = sys.argv[1]
    args = load_yaml_cfg(args_file)
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')
    args = init_featurizer(args)
    args['in_node_feats'] = args['node_featurizer'].feat_size()
    if args['edge_featurizer'] is not None:
        args['in_edge_feats'] = args['edge_featurizer'].feat_size()
    if 'task_names' in args:
        args['task_names'] = args['task_names'].split(',')
        ##args['n_tasks'] = len(args['task_names'])
    else:
        args['task_names'] = None
        args['n_tasks'] = 12
    eval_and_save(args)