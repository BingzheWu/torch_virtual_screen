from dgl.transform import add_self_loop
from dgllife.data import MoleculeCSVDataset, Tox21
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph, ScaffoldSplitter, RandomSplitter
from functools import partial

def load_dataset(args, df):
    """
    df: pd.dataframe loaded from smiles csv files
    """
    dataset = MoleculeCSVDataset(df=df,
    smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
    node_featurizer=args['node_featurizer'],
    edge_featurizer=args['edge_featurizer'],
    smiles_column=args['smiles_column'],
    cache_file_path=args['result_path']+'/graph.bin',
    task_names=args['task_names'],
    n_jobs=args['num_workers']
    )
    return dataset

def _test():
    import sys
    import yaml
    args_file = sys.argv[1]
    with open(args_file, 'r') as f:
        args = yaml.load(f)
    print(args)
    dataset = Tox21(smiles_to_graph=smiles_to_bigraph, cache_file_path='tox21_dglgraph.bin',load=True)
    g = dataset[0]
    

if __name__ == '__main__':
    _test()
