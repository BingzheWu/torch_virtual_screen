import numpy as np
from numpy.lib import split
from .train_utils import predict, load_model
import torch
import os
from .utils import load_yaml_cfg
from .dataset import dataset_loader, collate_molgraphs, split_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd

def eval_and_save(args, mode='test'):
    model = load_model(args)
    model_file = os.path.join(args['result_path'], 'model.pth')
    model = load_model(args)
    model.load_state_dict(torch.load(model_file)['model_state_dict'])
    dataset= dataset_loader(args, split=False)
    train_set, val_set, test_set = split_dataset(args, dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    print(dataset.df.columns)
    task_id = dataset.df.columns.get_loc(args['task_names'][0])
    print(task_id)
    if mode=='test':
        data_loader = test_loader
    predictions = np.zeros((len(data_loader.dataset),2))
    targets = np.zeros(len(data_loader.dataset))
        
    model.eval()
    k = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])[:,task_id-1]
            logits = predict(args, model, bg)[:,task_id-1]
            confidence = F.sigmoid(logits).view(-1, 1)
            predictions[k:(k+labels.size(0))] = np.concatenate([1-confidence.cpu().numpy(), confidence.cpu().numpy()], axis=1 )
            targets[k : (k + labels.size(0))] = labels.cpu().numpy()
            k += labels.shape[0]
        eps = 1e-12
        print(predictions.shape)
        print(targets.shape)
        entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
        print(entropies)
        print("Accuracy:", np.mean(np.argmax(predictions, axis=1) == targets))
        save_path = os.path.join(args['result_path'], args['dataset']+'_'+args['task_names'][0]+'test.npz')
        print(save_path)
        np.savez(save_path, entropies=entropies, predictions=predictions, targets=targets)

    
    