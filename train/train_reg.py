import sys
sys.path.append('./')
from core.train_utils import run_a_train_epoch, init_featurizer, load_model, mkdir_p, run_an_eval_epoch
from core.dataset import dataset_loader, collate_molgraphs, BalancedBatchSampler
import yaml
import torch
from torch.utils.data import DataLoader 
import torch.nn as nn
from torch.optim import Adam
from dgllife.utils import EarlyStopping, Meter

def main(args):
    train_set, val_set, test_set = dataset_loader(args)
    sampler = None
    shuffle = True
    #print(train_set[0])
    train_loader = DataLoader(dataset=train_set, sampler=sampler, shuffle=shuffle, batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    #print(len(train_loader))
    model = load_model(args)
    loss_criterion = nn.L1Loss(reduction='none')
    optimizer = Adam(model.parameters(), lr=args['lr'],
                         weight_decay=args['weight_decay'])
    stopper = EarlyStopping(patience=args['patience'],
                                filename=args['result_path'] + '/model.pth',
                                metric=args['metric'])
    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'],
            val_score, args['metric'], stopper.best_score))

        if early_stop:
            break

    if not args['pretrain']:
        stopper.load_checkpoint(model)
    val_score = run_an_eval_epoch(args, model, val_loader)
    test_score = run_an_eval_epoch(args, model, test_loader)
    print('val {} {:.4f}'.format(args['metric'], val_score))
    print('test {} {:.4f}'.format(args['metric'], test_score))

    with open(args['result_path'] + '/eval.txt', 'w') as f:
        if not args['pretrain']:
            f.write('Best val {}: {}\n'.format(args['metric'], stopper.best_score))
        f.write('Val {}: {}\n'.format(args['metric'], val_score))
        f.write('Test {}: {}\n'.format(args['metric'], test_score))
    
if __name__ =='__main__':
    arg_file = sys.argv[1]
    with open(arg_file, 'r') as f:
        args = yaml.load(f)
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')
    mkdir_p(args['result_path'])
    args = init_featurizer(args)
    args['in_node_feats'] = args['node_featurizer'].feat_size()
    if args['edge_featurizer'] is not None:
        args['in_edge_feats'] = args['edge_featurizer'].feat_size()
    main(args)