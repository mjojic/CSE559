'''
Part of catELMo
(c) 2023 by  Pengfei Zhang, Michael Cai, Seojin Bang, Heewook Lee, and Arizona State University.
See LICENSE-CC-BY-NC-ND for licensing.
'''

import sys
import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score
import torch
# from tqdm import tqdm
from datetime import datetime
from numpy import mean, std
import sklearn.model_selection
import lstm_utils as lstm
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'


def load_data_split(dat,split_type, seed):
    n_fold = 5
    idx_test_fold = 0
    idx_val_fold = -1
    idx_test = None
    idx_train = None
    x_pep = dat.epi
    x_tcr = dat.tcr
    
    if split_type == 'random':
        n_total = len(x_pep)
    elif split_type == 'epi':
        unique_peptides = np.unique(x_pep)
        n_total = len(unique_peptides)
    elif split_type == 'tcr':
        unique_tcrs = np.unique(x_tcr)
        n_total = len(unique_tcrs)
        
    np.random.seed(seed)    
    idx_shuffled = np.arange(n_total)
    np.random.shuffle(idx_shuffled)
    
    # Determine data split from folds
    n_test = int(round(n_total / n_fold))
    n_train = n_total - n_test

    # Determine position of current test fold
    test_fold_start_index = idx_test_fold * n_test
    test_fold_end_index = (idx_test_fold + 1) * n_test

    if split_type == 'random':
        # Split data evenly among evenly spaced folds
        # Determine if there is an outer testing fold
        if idx_val_fold < 0:
            idx_test = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_test = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'epi':
        if idx_val_fold < 0:
            idx_test_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_peptides = unique_peptides[idx_test_remove_pep]
            idx_test_pep = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_test_remove = [index for index, pep in enumerate(x_pep) if pep in test_remove_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'tcr':
        if idx_val_fold < 0:
            idx_test_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_tcrs = unique_tcrs[idx_test_remove_tcr]
            idx_test_tcr = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_test_remove = [index for index, tcr in enumerate(x_tcr) if tcr in test_remove_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)).difference(set(idx_test_remove)))

    testData = dat.iloc[idx_test, :].sample(frac=1).reset_index(drop=True)
    # print(testData.head(5))
    trainData = dat.iloc[idx_train, :].sample(frac=1).reset_index(drop=True)
    # print(trainData.head(5))
    

    print('================check Overlapping========================')
    print('number of overlapping tcrs: ', str(len(set(trainData.tcr).intersection(set(testData.tcr)))))
    print('number of overlapping epitopes: ', str(len(set(trainData.epi).intersection(set(testData.epi)))))
    
    # tcr_split testing read 
    X1_test_list, X2_test_list, y_test_list = testData.tcr_embeds.to_list(), testData.epi_embeds.to_list(),testData.binding.to_list()
    X1_test, X2_test, y_test = np.array(X1_test_list), np.array(X2_test_list), np.array(y_test_list)
    # tcr_split training read 
    X1_train_list, X2_train_list, y_train_list = trainData.tcr_embeds.to_list(), trainData.epi_embeds.to_list(),trainData.binding.to_list()
    X1_train, X2_train, y_train = np.array(X1_train_list), np.array(X2_train_list), np.array(y_train_list)
    return  X1_train, X2_train, y_train, X1_test, X2_test, y_test, testData, trainData

def lstm_get_lists_from_pairs(pairs):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label = pair
        tcrs.append(tcr)
        peps.append(pep[0])
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs

def train_(embedding_name, train_X, train_y, test_X, test_y, device:str = "cpu", epochs:int = 100, modified:bool = False):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    log_file = f"logs/{embedding_name}.log"
    output_file = f"outputs/{embedding_name}.txt"
    graph_file = f"graphs/{embedding_name}_metrics.png"
    model_file = f"models/{embedding_name}.pt"

    print(f"Modified: {modified}")
    with open(log_file, "a") as f:
        f.write(f"Beginning training of {embedding_name} at {datetime.now()}\n")
        f.write(f"Modified: {modified}\n")

    # Hyperparameters
    # hyper-params
    print("Setting up arg dict")
    arg = {}
    arg['train_auc_file'] = f"{embedding_name}_train_auc"
    arg['test_auc_file'] = f"{embedding_name}_test_auc"
    # if args.test_auc_file is None or args.test_auc_file == 'auto':
    #     dir = 'save_results'
    #     p_key = 'protein' if args.protein else ''
    #     arg['test_auc_file'] = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key])
    # arg['ae_file'] = args.ae_file
    # if args.ae_file is None or args.ae_file == 'auto':
    #     args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
    #     arg['ae_file'] = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
    #     pass
    arg["modified"] = modified
    arg['siamese'] = False
    print("Setting up params dict")
    params = {}
    params['lr'] = 1e-4
    params['wd'] = 0
    params['epochs'] = epochs
    # if args.dataset == 'tumor':
    #     params['epochs'] = 25
    params['batch_size'] = 50 # hardcoded into the predict function! no clue why!
    params['lstm_dim'] = 500
    params['emb_dim'] = 10
    params['dropout'] = 0.1
    params['option'] = 0
    params['enc_dim'] = 100
    params['train_ae'] = True

    print("Creating amino acid to index dict")
    # Used for converting amino acids to indices for the model. IDK if we can do without it or not, or replace it with something else.
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}

    # Define the model 
    # train
    with open(log_file, "a") as f:
        f.write(f"Creating train lists at {datetime.now()}\n")
    print("Creating train lists:\n\tTCRs")
    train_tcrs = train_X["tcr"].to_list()
    print("\tEpis")
    train_peps = train_X["epi"].to_list()
    print("\tSigns")
    train_signs = train_y.to_list()
    
    with open(log_file, "a") as f:
        f.write(f"Converting train lists at {datetime.now()}\n")
    print("Converting train data to lists of indices")
    lstm.convert_data(train_tcrs, train_peps, amino_to_ix) # Converts the strings into lists of indices

    with open(log_file, "a") as f:
        f.write(f"Getting train batches at {datetime.now()}\n")
    print("Getting train batches from converted train lists")
    train_batches = lstm.get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])

    # test
    # test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(testData)
    with open(log_file, "a") as f:
        f.write(f"Creating test lists at {datetime.now()}\n")
    print("Creating test lists:\n\tTCRs")
    test_tcrs = test_X["tcr"].to_list()
    print("\tEpis")
    test_peps = test_X["epi"].to_list()
    print("\tSigns")
    test_signs = test_y.to_list()

    with open(log_file, "a") as f:
        f.write(f"Converting test lists at {datetime.now()}\n")
    print("Converting test data to lists of indices")
    lstm.convert_data(test_tcrs, test_peps, amino_to_ix) # Converts the strings into lists of indices

    with open(log_file, "a") as f:
        f.write(f"Getting test batches at {datetime.now()}\n")
    print("Getting test batches from converted test lists")
    test_batches = lstm.get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])

    # Train the model
    with open(log_file, "a") as f:
        f.write(f"Beginning training at {datetime.now()}\n")
    print("Beginning training")
    model, metrics_dict = lstm.train_model(train_batches, test_batches, device, arg, params)

    # Save trained model
    with open(log_file, "a") as f:
        f.write(f"Saving model weights to {model_file} at {datetime.now()}\n")
    print(f"Saving model weights to {model_file}")
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'params': params
    #     }, 
    #     model_file)
    torch.save(model.state_dict(), model_file)

    with open(log_file, "a") as f:
        f.write(f"Metrics being generated at {output_file} at {datetime.now()}\n")
    
    print('================Performance of last epoch========================')
    
    print(f'{embedding_name} AUC: {metrics_dict["auc_score"][-1]}')

    print(f'precision_recall_fscore_macro is {metrics_dict["precision_recall_fscore_macro"][-1]}')
    print(f'acc is {metrics_dict["accuracy"][-1]}')
    print(f'precision1 is {metrics_dict["precision1"][-1]}')
    print(f'precision0 is {metrics_dict["precision0"][-1]}')
    print(f'recall1 is {metrics_dict["recall1"][-1]}')
    print(f'recall0 is {metrics_dict["recall0"][-1]}')
    print(f'f1macro is {metrics_dict["f1macro"][-1]}')
    print(f'f1micro is {metrics_dict["f1micro"][-1]}')

    with open(output_file, "a") as f:
        f.write(f"Generated results at {datetime.now()}\n")
        f.write("Showing last epoch's metrics\n")
        f.write("------------------------------------------\n")
        f.write(f"Modifed? {modified}\n")
        f.write(f'AUC: {metrics_dict["auc_score"][-1]}\n')
        f.write(f'Accuracy: {metrics_dict["accuracy"][-1]}\n')
        f.write(f'Precision Recall FScore Macro: {metrics_dict["precision_recall_fscore_macro"][-1]}\n')
        f.write(f'Precision 1: {metrics_dict["precision1"][-1]}\n')
        f.write(f'Precision 0: {metrics_dict["precision0"][-1]}\n')
        f.write(f'Recall 1: {metrics_dict["recall1"][-1]}\n')
        f.write(f'Recall 0: {metrics_dict["recall0"][-1]}\n')
        f.write(f'F1 Macro: {metrics_dict["f1macro"][-1]}\n')
        f.write(f'F1 Micro: {metrics_dict["f1micro"][-1]}\n')
        f.write("\n")

    # plotting code absolutely STOLEN from marko :P
    epoch_range = range(1, len(metrics_dict["accuracy"]) + 1)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 4, 1)
    plt.plot(epoch_range, metrics_dict["loss"], label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 4, 2)
    plt.plot(epoch_range, metrics_dict["accuracy"], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(3, 4, 3)
    plt.plot(epoch_range, metrics_dict["auc_score"], label='AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.subplot(3, 4, 4)
    plt.plot(epoch_range, metrics_dict["precision0"], label='Precision 0')
    plt.xlabel('Epochs')
    plt.ylabel('Precision 0')
    plt.legend()
    
    plt.subplot(3, 4, 5)
    plt.plot(epoch_range, metrics_dict["precision1"], label='Precision 1')
    plt.xlabel('Epochs')
    plt.ylabel('Precision 1')
    plt.legend()
    
    plt.subplot(3, 4, 6)
    plt.plot(epoch_range, metrics_dict["precision_recall_fscore_macro"], label='Precision Recall FScore Macro')
    plt.xlabel('Epochs')
    plt.ylabel('Precision Recall FScore Macro')
    plt.legend()

    plt.subplot(3, 4, 7)
    plt.plot(epoch_range, metrics_dict["recall0"], label='Recall 0')
    plt.xlabel('Epochs')
    plt.ylabel('Recall 0')
    plt.legend()

    plt.subplot(3, 4, 8)
    plt.plot(epoch_range, metrics_dict["recall1"], label='Recall 1')
    plt.xlabel('Epochs')
    plt.ylabel('Recall 1')
    plt.legend()

    plt.subplot(3, 4, 9)
    plt.plot(epoch_range, metrics_dict["f1macro"], label='F1 Macro')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Macro')
    plt.legend()

    plt.subplot(3, 4, 10)
    plt.plot(epoch_range, metrics_dict["f1micro"], label='F1 Micro')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Micro')
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig(graph_file)
    plt.close()

    with open(log_file, "a") as f:
        f.write(f"Metrics plot saved to {graph_file} at {datetime.now()}\n")
    print(f"Metrics plot saved to {graph_file}")

    with open(log_file, "a") as f:
        f.write(f"End of logs at {datetime.now()}\n")

def main(name, split="tcr",fraction=1.0, seed=42, device="cpu", epochs:int = 100, modified:bool = False):
    column_names = ["epi", "tcr", "binding"]
    print(f"Getting the data for the split {split}")
    if split == "epi":
        train = pd.read_csv("epi_split_train.csv", names=column_names, header=None)
        train = train.sample(random_state=seed, frac=fraction)
        test = pd.read_csv("epi_split_test.csv", names=column_names, header=None)
        test = test.sample(random_state=seed, frac=fraction)
    else:
        train = pd.read_csv("tcr_split_train.csv", names=column_names, header=None)
        train = train.sample(random_state=seed, frac=fraction)
        test = pd.read_csv("tcr_split_test.csv", names=column_names, header=None)
        test = test.sample(random_state=seed, frac=fraction)
    print(f"Preprocessing the data")
    train = train.astype({"binding": float})
    test = test.astype({"binding":float})

    train_X = train.drop(columns=['binding'])
    train_X["epi"] = train_X["epi"].apply(lambda epi: epi.upper().replace("O", "Q").replace("B", "G"))
    train_X["tcr"] = train_X["tcr"].apply(lambda tcr: tcr.upper().replace("O", "Q").replace("B", "G"))
    train_y = train["binding"]
    test_X = test.drop(columns=["binding"])
    test_X["epi"] = test_X["epi"].apply(lambda epi: epi.upper().replace("O", "Q").replace("B", "G"))
    test_X["tcr"] = test_X["tcr"].apply(lambda tcr: tcr.upper().replace("O", "Q").replace("B", "G"))
    test_y = test["binding"]

    embedding_name = f"{name}_{split}_seed_{seed}_fraction_{fraction}_modified_{modified}_epochs_{epochs}"
    print(f"Data preprocessing done, running model under the name {embedding_name}")

    train_(embedding_name, train_X, train_y, test_X, test_y, device=device, epochs=epochs, modified=modified)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,help='Name of the model (for output)')
    parser.add_argument('--split', type=str,help='tcr or epi')
    parser.add_argument('--device', type=str)
    parser.add_argument('--fraction', type=float, default=1.0) 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--modified", type=int)
    args = parser.parse_args()
    if args.modified is not None and args.modified > 0:
        modified = True
    else:
        modified = False
    if args.split is not None and args.split == "tcr":
        split = "tcr"
    else:
        split = "epi"
    if args.device is not None:
        device = args.device
    else:
        device = "cpu"
    if args.fraction is not None:
        fraction = args.fraction
    else:
        fraction = 1.0
    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1
    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = 1
    if args.name is not None:
        name = args.name
    else:
        name = "default_name"
    main(name=name, split=split, fraction=fraction, seed=seed, device=device, epochs=epochs, modified=modified)
