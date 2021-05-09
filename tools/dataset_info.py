import sys
sys.path.append('./')
import pandas as pd
import yaml
from core.dataset import dataset_loader
from core.train_utils import init_featurizer
from core.utils import load_args
import matplotlib.pyplot as plt

def _test(args):
    train_set, val_set, test_set = dataset_loader(args)
    train_df = train_set.dataset.df.iloc[test_set.indices,:]
    train_df.drop(columns=['smiles'])
    print(train_df)
    train_df = train_df.fillna('NAN')
    print(train_df)
    cols = ['NR-AR', 'NR-AR-LBD', 'NR-AhR']
    cols = train_df.columns[:6]
    fig, axes = plt.subplots(ncols=len(cols), figsize=(10,5))
    for col, ax in zip(train_df[cols], axes):
        print(train_df[col].value_counts())
        train_df[col].value_counts().plot.bar(ax=ax, title=col)
    plt.savefig('hist.pdf')
if __name__ == '__main__':
    args_file = sys.argv[1]
    args = load_args(args_file)
    _test(args)
