from dgllife.utils import Meter
import numpy as np
from torch.optim import SGD, Adam
import torch.nn as nn
import torch

def run_fedavg_epoch(args, model, data_loader):
    model.train()
    train_meter = Meter()
    loss_criterion = nn.SmoothL1Loss(reduction='none')
    if args['local_optimizer']=='ADAM':
        optimizer = Adam(model.parameters(), lr=args['lr'],
                         weight_decay=args['weight_decay'])
    else:
        optimizer = SGD(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    for id, data in enumerate(data_loader):
        smiles, bg, labels, masks = data
        if len(smiles) == 1:
            continue
        labels, masks = labels.to(args['device']), masks.to(args['device'])
        logits = predict(args, model, bg)
        loss = (loss_criterion(logits, labels)* (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def run_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            logits = predict(args, model, bg)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric']))
def predict(args, model, bg):
    bg = bg.to(args['device'])
    if args['edge_featurizer'] is None:
        node_feats = bg.ndata.pop('h').to(args['device'])
        return model(bg, node_feats)
    else:
        node_feats = bg.ndata.pop('h').to(args['device'])
        edge_feats = bg.edata.pop('e').to(args['device'])
        return model(bg, node_feats, edge_feats)
