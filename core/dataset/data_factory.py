from dgl.transform import add_self_loop
from dgllife.data import HIV
from dgllife.data import ESOL
from .tox21 import Tox21
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph, ScaffoldSplitter, RandomSplitter
import dgl
import torch
from functools import partial
def split_dataset(args, dataset):
    train_ratio, val_ratio, test_ratio = map(float, args['split_ratio'].split(','))
    if args['split'] == 'scaffold':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif args['split'] == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    else:
        return ValueError("Expect the splitting method to be 'scaffold', got {}".format(args['split']))

    return train_set, val_set, test_set
def dataset_loader(args, split=True):
    if args['dataset'] in ['Tox21']:
        dataset = Tox21(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True), 
            cache_file_path='tox21_dglgraph.bin',
            load=False,
            node_featurizer=args['node_featurizer'],
            edge_featurizer=args['edge_featurizer'],
            task_names=None,
            )
    if args['dataset'] == 'HIV':
        dataset = HIV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                      load=True,
                      node_featurizer=args['node_featurizer'],
                      edge_featurizer=args['edge_featurizer'],
                      n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    if args['dataset'] == 'ESOL':
        dataset = ESOL(partial(smiles_to_bigraph, add_self_loop=True), node_featurizer=args['node_featurizer'])
    train_set, val_set, test_set = split_dataset(args, dataset)
    if split:
        return train_set, val_set, test_set
    else:
        return dataset
def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks