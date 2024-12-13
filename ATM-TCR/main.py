import argparse
import os
import sys
import csv
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from data_loader import define_dataloader, load_embedding, load_data_split
from utils import str2bool, timeSince, get_performance_batchiter, print_performance, write_blackbox_output_batchiter, get_performance
import matplotlib.pyplot as plt
import data_io_tf
import torch.nn as nn

# Constants
PRINT_EVERY_EPOCH = 1

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
class CompoundLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(CompoundLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_loss = nn.BCELoss()
    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets) + self.bce_loss(inputs, targets)

def train(model, device, train_loader, optimizer, criterion, epoch):

    model.train()
    score = []
    label = []
    for batch in train_loader:

        x_pep, x_tcr, y = batch.X_pep.to(
            device), batch.X_tcr.to(device), batch.y.to(device)

        optimizer.zero_grad()
        yhat = model(x_pep, x_tcr)
        y = y.unsqueeze(-1).expand_as(yhat)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()

        score.extend(yhat.data.cpu().tolist())
        label.extend(y.data.cpu().tolist())

    perf = get_performance(score, label)

    print('[TRAIN] Epoch {} Loss {:.4f}'.format(epoch, loss.item()))
    print(perf)


def main():

    parser = argparse.ArgumentParser(description='Prediction of TCR binding to peptide-MHC complexes')

    parser.add_argument('--infile', type=str,
                        help='Input file for training')
    parser.add_argument('--indepfile', type=str, default=None,
                        help='Independent test data file')
    parser.add_argument('--blosum', type=str, default=None,
                        help='File containing BLOSUM matrix to initialize embeddings')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='Training batch size')
    parser.add_argument('--model_name', type=str, default='original.ckpt',
                        help = 'Model name to be saved/loaded for training/independent testing respectively')
    parser.add_argument('--epoch', type=int, default=200, metavar='N',
                        help='The maximum number of epochs to train')
    parser.add_argument('--min_epoch', type=int, default=30,
                        help='The minimum number of epochs to train, early stopping will not be applied until this epoch')
    parser.add_argument('--early_stop', type=str2bool, default=True,
                        help='Use early stopping method')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--cuda', type=str2bool, default=True,
                        help = 'enable cuda')
    parser.add_argument('--seed', type=int, default=1039,
                        help='random seed')
    parser.add_argument('--mode', type=str, default='train',
                        help = 'train or test')
    parser.add_argument('--save_model', type=str2bool, default=True,
                        help = 'save model')
    parser.add_argument('--model', type=str, default='attention',
                        help='Model to import')
    parser.add_argument('--drop_rate', type=float, default=0.25,
                        help='dropout rate')
    parser.add_argument('--lin_size', type=int, default=1024,
                        help='size of linear transformations')
    parser.add_argument('--padding', type=str, default='mid',
                        help='front, end, mid, alignment')
    parser.add_argument('--heads', type=int, default=5,
                        help='Multihead attention head')
    parser.add_argument('--max_len_tcr', type=int, default=20,
                        help='maximum TCR length allowed')
    parser.add_argument('--max_len_pep', type=int, default=22,
                        help='maximum peptide length allowed')
    parser.add_argument('--n_fold', type=int, default=5,
                        help='number of cross-validation folds')
    parser.add_argument('--idx_test_fold', type=int, default=0,
                        help='fold index for test set (0, ..., n_fold-1)')
    parser.add_argument('--idx_val_fold', type=int, default=-1,
                        help='fold index for validation set (-1, 0, ..., n_fold-1). \
                              If -1, the option will be ignored \
                              If >= 0, the test set will be set aside and the validation set is used as test set') 
    parser.add_argument('--split_type', type=str, default='random',
                        help='how to split the dataset (random, tcr, epitope)')
    parser.add_argument('--results_dir', type=str, default='/home/mjojic/CSE494/results_atm/plot.png',)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha value for focal loss')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='gamma value for focal loss')
    parser.add_argument('--compound_loss', type=str2bool, default=False,
                        help='use focal + BCE loss')
    parser.add_argument('--lr_schedule', type=int, default=7,
                        help='number of epochs before reducing lr')
    parser.add_argument('--lr_drop_factor', type = float, default=0.1,
                        help='factor to reduce lr by')
    parser.add_argument('--projection_dim', type=int, default=20,
                        help='dimension that the epitope/tcr will be projected to before dot product')
    parser.add_argument('--num_projections', type=int, default=10,
                        help='number of projections which will have dot product taken, and used as a single feature to classifier head')
    args = parser.parse_args()

    if args.mode == 'test':
        assert args.indepfile is not None, '--indepfile is missing!'
    assert args.idx_test_fold < args.n_fold, '--idx_test_fold should be smaller than --n_fold'
    assert args.idx_val_fold < args.n_fold, '--idx_val_fold should be smaller than --n_fold'
    assert args.idx_val_fold != args.idx_test_fold, '--idx_val_fold and --idx_test_fold should not be equal to each other'

    if args.compound_loss:
        print('Using Compound Loss')
        criterion = CompoundLoss(args.alpha, args.gamma)
    else:
        print('Using BCE Loss')
        criterion = F.binary_cross_entropy

    # Set Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load embedding matrix
    embedding_matrix = load_embedding(args.blosum)

    # Read data
    x_pep, x_tcr, y = data_io_tf.read_pTCR(args.infile)
    y = np.array(y)

    # Shuffle data into folds for cross validation
    idx_train, idx_test, idx_test_remove = load_data_split(x_pep, x_tcr, args)

    # Define dataloader
    train_loader = define_dataloader(x_pep[idx_train], x_tcr[idx_train], y[idx_train],
                                     args.max_len_pep, args.max_len_tcr,
                                     padding=args.padding,
                                     batch_size=args.batch_size, device=device)
    test_loader = define_dataloader(x_pep[idx_test], x_tcr[idx_test], y[idx_test],
                                    maxlen_pep=train_loader['pep_length'],
                                    maxlen_tcr=train_loader['tcr_length'],
                                    padding=args.padding,
                                    batch_size=args.batch_size, device=device)
    if args.indepfile is not None:
        x_indep_pep, x_indep_tcr, _ = data_io_tf.read_pTCR(args.indepfile)
        indep_loader = define_dataloader(x_indep_pep, x_indep_tcr, 
                                         maxlen_pep=train_loader['pep_length'],
                                         maxlen_tcr=train_loader['tcr_length'],
                                         padding=args.padding,
                                         batch_size=args.batch_size, device=device)

    args.pep_length = train_loader['pep_length']
    args.tcr_length = train_loader['tcr_length']

    # Define model
    if args.model == 'attention':
        from attention import Net
    elif args.model == 'cross_attention':
        print('Using Cross Attention Model')
        from attention import CrossAttentionNet as Net
    elif args.model == 'project_mult_model':
        print('Using model to project epi/tcr, then dot product to get features for classification')
        from attention import PepEpiMultNet as Net
    else:
        raise ValueError('unknown model name')
    print(f'Min epoch: {args.min_epoch}')

    model = Net(embedding_matrix, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create Required Directories
    if 'models' not in os.listdir('.'):
        os.mkdir('models')
    if 'result' not in os.listdir('.'):
        os.mkdir('result')

    # eax1it model
    if args.mode == 'train':
        wf_open = open(
            'result/perf_' + os.path.splitext(os.path.basename(args.model_name))[0] + '.csv', 'w')
        wf_colnames = ['loss', 'accuracy',
                       'precision1', 'precision0',
                       'recall1', 'recall0',
                       'f1macro', 'f1micro', 'auc']
        wf = csv.DictWriter(wf_open, wf_colnames, delimiter='\t')

        losses = []
        accuracies = []
        precisions1 = []
        precisions0 = []
        recalls1 = []
        recalls0 = []
        f1macros = []
        f1micros = []
        aucs = []

        max_acc = []
        max_prec1 = []
        max_prec0 = []
        max_rec1 = []
        max_rec0 = []
        max_f1macro = []
        max_f1micro = []
        max_auc = []

        n_epochs = args.epoch

        t0 = time.time()
        lossArraySize = 10
        lossArray = deque([sys.maxsize], maxlen=lossArraySize)
        lr = args.lr
        lr_schedule = args.lr_schedule
        drop_factor = args.lr_drop_factor
        for epoch in range(1, args.epoch + 1):
            if epoch % lr_schedule == 0:
                lr *= drop_factor
                optimizer = optim.Adam(model.parameters(), lr=lr)

            train(model, device, train_loader['loader'], optimizer, criterion, epoch)
            
            # Print performance
            print('[TEST ] {} ----------------'.format(epoch))
            perf_test = get_performance_batchiter(
                test_loader['loader'], model, device)
            print(perf_test)
            losses.append(perf_test['loss'])
            accuracies.append(perf_test['accuracy'])
            precisions1.append(perf_test['precision1'])
            precisions0.append(perf_test['precision0'])
            recalls1.append(perf_test['recall1'])
            recalls0.append(perf_test['recall0'])
            f1macros.append(perf_test['f1macro'])
            f1micros.append(perf_test['f1micro'])
            aucs.append(perf_test['auc'])

            if len(max_acc) != 0:
                max_auc.append(max(perf_test['auc'], max_auc[-1]))
                max_acc.append(max(perf_test['accuracy'], max_acc[-1]))
                max_prec1.append(max(perf_test['precision1'], max_prec1[-1]))
                max_prec0.append(max(perf_test['precision0'], max_prec0[-1]))
                max_rec1.append(max(perf_test['recall1'], max_rec1[-1]))
                max_rec0.append(max(perf_test['recall0'], max_rec0[-1]))
                max_f1macro.append(max(perf_test['f1macro'], max_f1macro[-1]))
                max_f1micro.append(max(perf_test['f1micro'], max_f1micro[-1]))
            else:
                max_auc.append(perf_test['auc'])
                max_acc.append(perf_test['accuracy'])
                max_prec1.append(perf_test['precision1'])
                max_prec0.append(perf_test['precision0'])
                max_rec1.append(perf_test['recall1'])
                max_rec0.append(perf_test['recall0'])
                max_f1macro.append(perf_test['f1macro'])
                max_f1micro.append(perf_test['f1micro'])

            

            # Check for early stopping
            lossArray.append(perf_test['loss'])
            average_loss_change = sum(np.abs(np.diff(lossArray))) / lossArraySize
            if epoch > args.min_epoch and average_loss_change < 10 and args.early_stop:
                print('Early stopping at epoch {}'.format(epoch))
                n_epochs = epoch
                break

        epochs = range(1, n_epochs + 1)

        
        plt.figure(figsize=(15, 20))

        # Plotting the metrics over epochs
        plt.subplot(6, 3, 1)
        plt.plot(epochs, losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(6, 3, 2)
        plt.plot(epochs, accuracies, label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(6, 3, 3)
        plt.plot(epochs, precisions1, label='Precision1')
        plt.xlabel('Epochs')
        plt.ylabel('Precision1')
        plt.legend()

        plt.subplot(6, 3, 4)
        plt.plot(epochs, precisions0, label='Precision0')
        plt.xlabel('Epochs')
        plt.ylabel('Precision0')
        plt.legend()

        plt.subplot(6, 3, 5)
        plt.plot(epochs, recalls1, label='Recall1')
        plt.xlabel('Epochs')
        plt.ylabel('Recall1')
        plt.legend()

        plt.subplot(6, 3, 6)
        plt.plot(epochs, recalls0, label='Recall0')
        plt.xlabel('Epochs')
        plt.ylabel('Recall0')
        plt.legend()

        plt.subplot(6, 3, 7)
        plt.plot(epochs, f1macros, label='F1 Macro')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Macro')
        plt.legend()

        plt.subplot(6, 3, 8)
        plt.plot(epochs, f1micros, label='F1 Micro')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Micro')
        plt.legend()

        plt.subplot(6, 3, 9)
        plt.plot(epochs, aucs, label='AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()

        # Plotting the maximum values
        plt.subplot(6, 3, 10)
        plt.plot(epochs, [max(accuracies)] * len(epochs), 'r--', label=f'Max Accuracy: {max(accuracies):.4f}')
        plt.plot(epochs, max_acc, 'g', label='Upper Limit')
        plt.xlabel('Epochs')
        plt.ylabel('Max Accuracy')
        plt.legend()

        plt.subplot(6, 3, 11)
        plt.plot(epochs, [max(precisions1)] * len(epochs), 'r--', label=f'Max Precision1: {max(precisions1):.4f}')
        plt.plot(epochs, max_prec1, 'g', label='Upper Limit')
        plt.xlabel('Epochs')
        plt.ylabel('Max Precision1')
        plt.legend()

        plt.subplot(6, 3, 12)
        plt.plot(epochs, [max(precisions0)] * len(epochs), 'r--', label=f'Max Precision0: {max(precisions0):.4f}')
        plt.plot(epochs, max_prec0, 'g', label='Upper Limit')
        plt.xlabel('Epochs')
        plt.ylabel('Max Precision0')
        plt.legend()

        plt.subplot(6, 3, 13)
        plt.plot(epochs, [max(recalls1)] * len(epochs), 'r--', label=f'Max Recall1: {max(recalls1):.4f}')
        plt.plot(epochs, max_rec1, 'g', label='Upper Limit')
        plt.xlabel('Epochs')
        plt.ylabel('Max Recall1')
        plt.legend()

        plt.subplot(6, 3, 14)
        plt.plot(epochs, [max(recalls0)] * len(epochs), 'r--', label=f'Max Recall0: {max(recalls0):.4f}')
        plt.plot(epochs, max_rec0, 'g', label='Upper Limit')
        plt.xlabel('Epochs')
        plt.ylabel('Max Recall0')
        plt.legend()

        plt.subplot(6, 3, 15)
        plt.plot(epochs, [max(f1macros)] * len(epochs), 'r--', label=f'Max F1 Macro: {max(f1macros):.4f}')
        plt.plot(epochs, max_f1macro, 'g', label='Upper Limit')
        plt.xlabel('Epochs')
        plt.ylabel('Max F1 Macro')
        plt.legend()

        plt.subplot(6, 3, 16)
        plt.plot(epochs, [max(f1micros)] * len(epochs), 'r--', label=f'Max F1 Micro: {max(f1micros):.4f}')
        plt.plot(epochs, max_f1micro, 'g', label='Upper Limit')
        plt.xlabel('Epochs')
        plt.ylabel('Max F1 Micro')
        plt.legend()

        plt.subplot(6, 3, 17)
        plt.plot(epochs, [max(aucs)] * len(epochs), 'r--', label=f'Max AUC: {max(aucs):.4f}')
        plt.plot(epochs, max_auc, 'g', label='Upper Limit')
        plt.xlabel('Epochs')
        plt.ylabel('Max AUC')
        plt.legend()

        plt.tight_layout()
        plt.savefig(args.results_dir)
        plt.close()

        print(os.path.splitext(os.path.basename(args.model_name))[0])
        print(timeSince(t0))

        # evaluate and print independent-test-set performance
        if args.indepfile is not None:
            print('[INDEP] {} ----------------')
            perf_indep = get_performance_batchiter(
                indep_loader['loader'], model, device)

            wf_open = open('result/perf_' + os.path.splitext(os.path.basename(args.model_name))[0] + '_' +
                           os.path.basename(args.indepfile), 'w')
            wf = csv.DictWriter(wf_open, wf_colnames, delimiter='\t')
            print_performance(perf_indep, writeif=True, wf=wf)

            wf_open1 = open('data/pred_' + os.path.splitext(os.path.basename(args.model_name))[0] + '_' +
                            os.path.basename(args.indepfile), 'w')
            wf1 = csv.writer(wf_open1, delimiter='\t')
            write_blackbox_output_batchiter(
                indep_loader, model, wf1, device, ifscore=True)

        # evaluate and print test-set performance
        print('[TEST ] {} ----------------'.format(epoch))
        perf_test = get_performance_batchiter(
            test_loader['loader'], model, device)
        print_performance(perf_test)

        if args.save_model:

            wf_open1 = open(
                'result/pred_' + os.path.splitext(os.path.basename(args.model_name))[0] + '.csv', 'w')
            wf1 = csv.writer(wf_open1, delimiter='\t')
            write_blackbox_output_batchiter(
                test_loader, model, wf1, device, ifscore=True)

            model_name = './models/' + \
                os.path.splitext(os.path.basename(args.model_name))[0] + '.ckpt'
            torch.save(model.state_dict(), model_name)
    
    elif args.mode == 'test':

        model_name = args.model_name

        assert model_name in os.listdir('./models')

        # model_name = './models/' + model_name
        model.load_state_dict(torch.load('best_model_projection_net.pth', map_location=torch.device(device)))

        # evaluate and print independent-test-set performance
        print('[INDEP] {} ----------------')
        perf_indep = get_performance_batchiter(
            indep_loader['loader'], model, device)
        print_performance(perf_indep)

        # write blackbox output
        wf_bb_open1 = open('result/pred_' + os.path.splitext(os.path.basename('model_name'))[0] + '_' +
                           os.path.basename(args.indepfile), 'w')
        wf_bb1 = csv.writer(wf_bb_open1, delimiter='\t')
        write_blackbox_output_batchiter(
            indep_loader, model, wf_bb1, device, ifscore=False)

    else:
        print('\nError: "--mode train" or "--mode test" expected')

if __name__ == '__main__':
    main()
