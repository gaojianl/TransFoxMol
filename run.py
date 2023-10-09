import torch, math, argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from torch.nn import BCELoss
from torch_geometric.data import DataLoader
from TFM.Dataset import MolNet
from TFM.model import Fox
from TFM.utils import get_logger, metrics_c, metrics_r, set_seed, load_data, LogCoshLoss
from rdkit.Chem.SaltRemover import SaltRemover
import hyperopt
from hyperopt import fmin, hp, Trials
from hyperopt.early_stop import no_progress_loss
import warnings
warnings.filterwarnings("ignore")
remover = SaltRemover()
bad = ['He', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Gd', 'Tb', 'Ho', 'W', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac']
_use_shared_memory = True
torch.backends.cudnn.benchmark = True


def training(model, train_loader, optimizer, loss_f, metric, task, device, mean, stds):
    loss_record, record_count = 0., 0.
    preds = torch.Tensor([]); tars = torch.Tensor([])
    model.train()
    if task == 'clas':
        for data in train_loader:
            if data.y.size()[0] > 1:
                y = data.y.to(device)
                logits = model(data)
                
                loss = loss_f(logits.squeeze(), y.squeeze())
                loss_record += float(loss.item())
                record_count += 1
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
                optimizer.step()

                pred = logits.detach().cpu()
                preds = torch.cat([preds, pred], 0); tars = torch.cat([tars, y.cpu()], 0)
        clas = preds > 0.5
        acc, f1, pre, rec, auc = metric(clas.squeeze().numpy(), preds.squeeze().numpy(), tars.squeeze().numpy())
    else:
        for data in train_loader:
            if data.y.size()[0] > 1:
                y = data.y.to(device)
                
                y_ = (y - mean) / (stds+1e-5)
                logits = model(data)
                
                loss = loss_f(logits.squeeze(), y_.squeeze())
                loss_record += float(loss.item())
                record_count += 1
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
                optimizer.step()

                pred = logits.detach().cpu()
                pred = pred*stds+mean
                preds = torch.cat([preds, pred], 0); tars = torch.cat([tars, y.cpu()], 0)
        acc, f1, pre, rec, auc = metric(preds.squeeze().numpy(), tars.squeeze().numpy())

    epoch_loss = loss_record / record_count
    return epoch_loss, acc, f1, pre, rec, auc


def testing(model, test_loader, loss_f, metric, task, device, mean, stds, resu):
    loss_record, record_count = 0., 0.
    preds = torch.Tensor([]); tars = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        if task == 'clas':
            for data in test_loader:
                if data.y.size()[0] > 1:
                    y = data.y.to(device)
                    logits = model(data)
                    
                    loss = loss_f(logits.squeeze(), y.squeeze())
                    loss_record += float(loss.item())
                    record_count += 1

                    pred = logits.detach().cpu() 
                    preds = torch.cat([preds, pred], 0); tars = torch.cat([tars, y.cpu()], 0)
            clas = preds > 0.5
            acc, f1, pre, rec, auc = metric(clas.squeeze().numpy(), preds.squeeze().numpy(), tars.squeeze().numpy())
        else:
            for data in test_loader:
                if data.y.size()[0] > 1:
                    y = data.y.to(device)

                    y_ = (y - mean) / (stds+1e-5)
                    logits = model(data)
                    
                    loss = loss_f(logits.squeeze(), y_.squeeze())
                    loss_record += float(loss.item())
                    record_count += 1

                    pred = logits.detach().cpu()
                    pred = pred*stds+mean
                    preds = torch.cat([preds, pred], 0); tars = torch.cat([tars, y.cpu()], 0)
            acc, f1, pre, rec, auc = metric(preds.squeeze().numpy(), tars.squeeze().numpy())

    epoch_loss = loss_record / record_count
    if resu:
        return epoch_loss, acc, f1, pre, rec, auc, preds, tars
    else:
        return epoch_loss, acc, f1, pre, rec, auc


def main(tasks, task, dataset, device, train_epoch, seed, fold, batch_size, rate, scaffold, modelpath, logger, lr, attn_head, output_dim, attn_layers, dropout, mean, stds, D, met, savem):
    logger.info('Dataset: {}  task: {}  train_epoch: {}'.format(dataset, task, train_epoch))
    d_k, seed_ = round(output_dim/attn_head), seed

    fold_result = [[], []]
    if task == 'clas':
        loss_f = BCELoss().to(device)
        metric = metrics_c(accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

        for fol in range(1, fold+1):
            best_val_auc, best_test_auc = 0., 0.
            if seed is not None:
                seed_ = seed + fol-1
                set_seed(seed_)
            model = Fox(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

            train_loader, valid_loader, test_loader = load_data(dataset, batch_size, rate[0], rate[1], 0, task, scaffold, seed_)
            logger.info('Dataset: {}  Fold: {:<4d}'.format(moldata, fol))

            for i in range(1,train_epoch+1):
                train_loss, train_acc, train_f1, train_pre, train_rec, train_auc = training(model, train_loader, optimizer, loss_f, metric, task, device, mean, stds)
                valid_loss, valid_acc, valid_f1, valid_pre, valid_rec, valid_auc = testing(model, valid_loader, loss_f, metric, task, device, mean, stds, False)
                
                logger.info('Dataset: {}  Epoch: {:<3d}  train_loss: {:.4f}  train_acc: {:.4f}  train_f1: {:.4f}  train_auc: {:.4f}  train_pre: {:.4f}  train_rec: {:.4f}'.format(dataset ,i, train_loss, train_acc, train_f1, train_auc, train_pre, train_rec))
                logger.info('Dataset: {}  Epoch: {:<3d}  valid_loss: {:.4f}  valid_acc: {:.4f}  valid_f1: {:.4f}  valid_auc: {:.4f}  valid_pre: {:.4f}  valid_rec: {:.4f}'.format(dataset, i, valid_loss, valid_acc, valid_f1, valid_auc, valid_pre, valid_rec))

                if valid_auc > best_val_auc:
                    best_val_auc = valid_auc
                    if savem:
                        model_save_path = modelpath + '{}_{}_{}.pkl'.format(dataset, i, round(valid_auc, 4))
                        torch.save(model.state_dict(), model_save_path)
                    test_loss, test_acc, test_f1, test_pre, test_rec, test_auc = testing(model, test_loader, loss_f, metric, task, device, mean, stds, False)
                    logger.info('Dataset: {}  Epoch: {:<3d}  test__loss: {:.4f}  test__acc: {:.4f}  test__f1: {:.4f}  test__auc: {:.4f}  test__pre: {:.4f}  test__rec: {:.4f}'.format(dataset, i, test_loss, test_acc, test_f1, test_auc, test_pre, test_rec))
                    best_test_auc = test_auc
            fold_result[0].append(best_val_auc)
            fold_result[1].append(best_test_auc)
            logger.info('Dataset: {} Fold: {} best_val_auc: {:.4f}  best_test_auc: {:.4f}'.format(dataset, fol, best_val_auc, best_test_auc))
        logger.info('Dataset: {} Fold result: {}'.format(dataset, fold_result))
        return fold_result

    else:
        loss_f = LogCoshLoss().to(device)
        metric = metrics_r(mean_absolute_error, mean_squared_error, r2_score)

        for fol in range(1, fold+1):
            best_val_rmse, best_test_rmse = 9999., 9999.
            if seed is not None:
                seed_ = seed + fol-1
                set_seed(seed_)
            model = Fox(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

            train_loader, valid_loader, test_loader = load_data(dataset, batch_size, rate[0], rate[1], 0, task, scaffold, seed_)
            logger.info('Dataset: {}  Fold: {:<4d}'.format(moldata, fol))

            for i in range(1, train_epoch+1):
                train_loss, train_mae, train_rmse, train_r2, _, _ = training(model, train_loader, optimizer, loss_f, metric, task, device, mean, stds)
                valid_loss, valid_mae, valid_rmse, valid_r2, _, _ = testing(model, valid_loader, loss_f, metric, task, device, mean, stds, False)
                logger.info('Dataset: {}  Epoch: {:<3d}  train_loss: {:.4f}  train_mae: {:.4f}  train_rmse: {:.4f} train_r2: {:.4f}'.format(dataset, i, train_loss, train_mae, train_rmse, train_r2))
                logger.info('Dataset: {}  Epoch: {:<3d}  valid_loss: {:.4f}  valid_mae: {:.4f}  valid_rmse: {:.4f} valid_r2: {:.4f}'.format(dataset, i, valid_loss, valid_mae, valid_rmse, valid_r2))

                if met == 'rmse':
                    if valid_rmse < best_val_rmse:
                        best_val_rmse = valid_rmse
                        if savem:
                            model_save_path = modelpath + '{}_{}_{}.pkl'.format(dataset, i, round(valid_rmse,4))
                            torch.save(model.state_dict(), model_save_path)
                        test_loss, test_mae, test_rmse, test_r2, _, _ = testing(model, test_loader, loss_f, metric, task, device, mean, stds, False)
                        logger.info('Dataset: {}  Epoch: {:<3d}  test_loss: {:.4f}  test_mae: {:.4f}  test_rmse: {:.4f} test_r2: {:.4f}'.format(dataset, i, test_loss, test_mae, test_rmse, test_r2))
                        best_test_rmse = test_rmse

                elif met == 'mae':
                    if valid_mae < best_val_rmse:
                        best_val_rmse = valid_mae
                        if savem:
                            model_save_path = modelpath + '{}_{}_{}.pkl'.format(dataset, i, round(valid_rmse,4))
                            torch.save(model.state_dict(), model_save_path)
                        test_loss, test_mae, test_rmse, test_r2, _, _ = testing(model, test_loader, loss_f, metric, task, device, mean, stds, False)
                        logger.info('Dataset: {}  Epoch: {:<3d}  test_loss: {:.4f}  test_mae: {:.4f}  test_rmse: {:.4f} test_r2: {:.4f}'.format(dataset, i, test_loss, test_mae, test_rmse, test_r2))
                        best_test_rmse = test_mae
                else:
                    raise ValueError('regression metric must be rmse or mae')
            fold_result[0].append(best_val_rmse)
            fold_result[1].append(best_test_rmse)
            logger.info('Dataset: {} Fold: {} best_val_{}: {:.4f}  best_test_{}: {:.4f}'.format(dataset, fol, met, best_val_rmse, met, best_test_rmse))
        logger.info('Dataset: {} Fold result: {}'.format(dataset, fold_result))
        return fold_result


def test(tasks, task, dataset, device, seed, batch_size, logger, attn_head, output_dim, attn_layers, dropout, pretrain, mean, stds, D):
    logger.info('Dataset: {}  task: {}  testing:'.format(dataset, task))
    d_k = round(output_dim/attn_head)
    if seed is not None:
        set_seed(seed)
    model = Fox(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device).to(device)
    state_dict = torch.load(pretrain)
    model.load_state_dict(state_dict)

    data = MolNet(root='./dataset', dataset=dataset)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)
    if task == 'clas':
        loss_f = BCELoss().to(device)
        metric = metrics_c(accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
        loss, acc, f1, pre, rec, auc, preds, tars = testing(model, loader, loss_f, metric, task, device, mean, stds, True)
        logger.info('Dataset: {}  test_loss: {:.4f}  test_acc: {:.4f}  test_f1: {:.4f}  test_auc: {:.4f}  test_pre: {:.4f}  test_rec: {:.4f}'.format(dataset, loss, acc, f1, auc, pre, rec))
        results = {
            'test_loss': loss,
            'test_acc': acc,
            'test_f1': f1,
            'test_pre': pre,
            'test_rec': rec,
            'test_auc': auc,
            'prediction': preds,
            'target': tars
        }
        np.save('log/Result'+moldata+'_test.npy', results, allow_pickle=True)
    else:
        loss_f = LogCoshLoss().to(device)
        metric = metrics_r(mean_absolute_error, mean_squared_error, r2_score)
        loss, mae, rmse, r2, _, _, preds, tars= testing(model, loader, loss_f, metric, task, device, mean, stds, True)
        logger.info('Dataset: {}  test_loss: {:.4f}  test_mae: {:.4f}  test_rmse: {:.4f} test_r2: {:.4f}'.format(dataset, loss, mae, rmse, r2))
        results = {
            'test_loss': loss,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_r2': r2,
            'prediction': preds,
            'target': tars
        }
        np.save('log/Result'+moldata+'_test.npy', results, allow_pickle=True)

def psearch(params):
    logger.info('Optimizing Hyperparameters')
    fold_result = main(params['tasks'],params['task'],params['moldata'],params['device'],params['train_epoch'],params['seed'],params['fold'],params['batch_size'],params['rate'],params['scaffold'],params['modelpath'],params['logger'],params['lr'],params['attn_head'],params['output_dim'],params['attn_layers'],params['dropout'],params['mean'], params['std'], params['D'], params['metric'], False)
    if task == 'reg':
        valid_res = np.mean(fold_result[1])
    else:
        valid_res = -np.mean(fold_result[1])
    return valid_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransFoxMol')
    parser.add_argument('mode', type=str, choices=['train', 'test', 'search'], help='train, test or hyperparameter_search')
    parser.add_argument('moldata', type=str, help='Dataset name')
    parser.add_argument('--task', type=str, choices=['clas', 'reg'], help='Classification or Regression')
    parser.add_argument('--device', type=str, default='cuda:0', help='Which gpu to use if any (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
    parser.add_argument('--train_epoch', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--valrate', type=float, default=0.1, help='valid rate (default: 0.1)')
    parser.add_argument('--testrate', type=float, default=0.1, help='test rate (default: 0.1)')
    parser.add_argument('--fold', type=int, default=3, help='Number of folds for cross validation (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout ratio')
    parser.add_argument('--scaffold', type =str, default=True, help = "True: random scaffold dataset split; False: random dataset split (default: True)")
    parser.add_argument('--attn_head', type = int, default=6, help = "Number of attention heads for transformer")
    parser.add_argument('--attn_layers', type=int, default=2, help='Number of GNN message passing layers')
    parser.add_argument('--output_dim', type=int, default=256, help='Hidden size of embedding layer')
    parser.add_argument('--D', type=int, default=4, help='Hidden size of readout layer')
    parser.add_argument('--seed', type=int, help = "Seed for splitting the dataset")
    parser.add_argument('--pretrain', type=str, help = "Path of retrained weights")
    parser.add_argument('--metric', type=str, choices=['rmse', 'mae'], help='Metric to evaluate the regression performance')
    args = parser.parse_args()

    device = torch.device(args.device)
    moldata = args.moldata
    rate = [args.valrate, args.testrate]

    if moldata in ['esol', 'freesolv', 'lipo', 'qm7']:
        task = 'reg'
        numtasks = 1
        if moldata == 'esol':
            labell = ['measured log solubility in mols per litre']
            mean, std = [-3.0493801421817386, 2.0991245476623708]
        elif moldata == 'freesolv':
            labell = ['expt']
            mean, std = [-3.8030062305295944, 3.844822204602953]
        elif moldata == 'lipo':
            labell = ['exp']
            mean, std = [2.1840152452444053, 1.2001577366239675]
        elif moldata == 'qm7':
            labell = ['u0_atom']
            mean, std = [-1544.8360893118595, 222.8738916827154]
        else:pass

    elif moldata in ['bbbp', 'sider', 'clintox', 'tox21', 'bace']:
        task = 'clas'
        mean, std = None, None
        if moldata == 'sider':
            numtasks = 27
            labell = ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders','Investigations', 'Musculoskeletal and connective tissue disorders', 
                      'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders','Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                      'General disorders and administration site conditions','Endocrine disorders', 'Surgical and medical procedures','Vascular disorders', 'Blood and lymphatic system disorders',
                      'Skin and subcutaneous tissue disorders','Congenital, familial and genetic disorders','Infections and infestations','Respiratory, thoracic and mediastinal disorders','Psychiatric disorders', 
                      'Renal and urinary disorders','Pregnancy, puerperium and perinatal conditions','Ear and labyrinth disorders', 'Cardiac disorders','Nervous system disorders','Injury, poisoning and procedural complications']
        elif moldata == 'clintox':
            numtasks = 2
            labell = ['FDA_APPROVED', 'CT_TOX']
        elif moldata == 'tox21':
            numtasks = 12
            labell = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        elif moldata == 'bace':
            numtasks = 1
            labell = ['Class']
        elif moldata == 'bbbp':
            numtasks = 1
            labell = ['p_np']
        else:pass
    else:
        mean, std = 0, 1
        task = args.task
        if args.task == 'reg':
            labell = ['standard_value']
        else:
            numtasks = 1
            labell = ['label']
    
    logf = 'log/{}_{}_{}.log'.format(moldata, args.task, args.mode)
    modelpath = 'log/checkpoint/'
    logger = get_logger(logf)
    
    moldata += task
    if args.mode == 'search':
        trials = Trials()
        if args.moldata in ['tox21', 'lipo', 'qm7']:
            batch_size = 256
        else:
            batch_size = 32
        if args.moldata == 'tox21':
            lrs = [1e-2, 5e-3, 1e-3]
        else:
            lrs = [1e-3, 5e-4, 1e-4]

        parm_space = {      # search space of param
            'tasks': numtasks,
            'task': task,
            'moldata': moldata,
            'mean': mean,
            'std': std,
            'device': args.device,
            'modelpath': modelpath,
            'logger': logger,
            'seed': args.seed,
            'fold': args.fold,
            'metric': args.metric,
            'rate': rate,
            'scaffold': args.scaffold,
            'train_epoch': args.train_epoch,
            'attn_head': hp.choice('attn_head', [4, 6, 8, 10]),
            'output_dim': hp.choice('output_dim', [128, 256]),
            'attn_layers': hp.choice('attn_layers', [1, 2, 3, 4]),
            'dropout': hp.choice('dropout', [0.05, 0.1]),
            'lr': hp.choice('lr', lrs),
            'D': hp.choice('D', [2, 4, 6, 8, 12, 16]),
            'batch_size': batch_size
            }
        param_mappings = {
            'attn_head': [4, 6, 8, 10],
            'output_dim': [128, 256],
            'attn_layers': [1, 2, 3, 4],
            'dropout': [0.05, 0.1],
            'lr': lrs,  # Placeholder for lr values. Please replace with actual values before running.
            'D': [2, 4, 6, 8, 12, 16]
            }
        best = fmin(fn=psearch, space=parm_space, algo=hyperopt.tpe.suggest, max_evals=100, trials=trials, early_stop_fn=no_progress_loss(50))
        best_values = {k: param_mappings[k][v] if k in param_mappings else v for k, v in best.items()}
        logger.info('Dataset {} Best Params: {}'.format(moldata, best_values))
        ys = [t['result']['loss'] for t in trials.trials]
        logger.info('Dataset {} Hyperopt Results: {}'.format(moldata, ys))

    elif args.mode == 'train':
        logger.info('Training')
        fold_result = main(numtasks, task, moldata, device, args.train_epoch, args.seed, args.fold, args.batch_size, rate,args.scaffold, modelpath, logger, args.lr, args.attn_head, args.output_dim, args.attn_layers, args.dropout, mean, std, args.D, args.metric, True)
        print(fold_result)
    elif args.mode == 'test':
        assert (args.pretrain is not None)
        fold_result = test(numtasks, task, moldata, device, args.seed, args.batch_size, logger, args.attn_head, args.output_dim, args.attn_layers, args.dropout, args.pretrain, mean, std, args.D)
    else:pass
