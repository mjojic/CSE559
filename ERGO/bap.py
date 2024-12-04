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
    log_file = f"logs/{embedding_name}.log"
    output_file = f"outputs/{embedding_name}.txt"

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
    # train_tcrs, train_peps, train_signs = lstm_get_lists_from_pairs(trainData)
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
    model, best_auc, best_roc = lstm.train_model(train_batches, test_batches, device, arg, params)
    # model.fit([X1_train,X2_train], y_train, verbose=0, validation_split=0.20, epochs=200, batch_size = 32)
    # model.save('models/' + embedding_name + '.hdf5')

    # Save trained model
    model_filepath = f"models/{embedding_name}.pt"
    with open(log_file, "a") as f:
        f.write(f"Saving model weights to {model_filepath} at {datetime.now()}\n")
    print(f"Saving model weights to {model_filepath}")
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'params': params
    #     }, 
    #     args.model_file)
    torch.save(model.state_dict(), model_filepath)
    # if args.roc_file:
    #     # Save best ROC curve and AUC
    #     np.savez(args.roc_file, fpr=best_roc[0], tpr=best_roc[1], auc=np.array(best_auc))

    # Predict with the models
    # yhat = model.predict([X1_test, X2_test])
    # device = os.environ["CUDA_VISIBLE_DEVICES"]
    
    # yhat = lstm.predict(model, test_batches, device)
    with open(log_file, "a") as f:
        f.write(f"Getting predictions from model at {datetime.now()}\n")
    print("Getting predictions")
    model.eval()
    yhat = []
    test_y_from_batch = []
    for batch in test_batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        prediction_tensor = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        yhat.extend(prediction_tensor.cpu().data.numpy())
        test_y_from_batch.extend(np.array(batch_signs))
    yhat = pd.DataFrame(yhat)
    test_y = pd.DataFrame(test_y_from_batch)

    with open(log_file, "a") as f:
        f.write(f"Metrics being generated at {output_file} at {datetime.now()}. End of logging\n")
    
    print('================Performance========================')
    
    auc_score = roc_auc_score(test_y, yhat)
    print(embedding_name+' AUC: ' + str(auc_score))
    
    yhat[yhat>=0.5] = 1
    yhat[yhat<0.5] = 0
    
    accuracy = accuracy_score(test_y, yhat)
    precision1 = precision_score(
        test_y, yhat, pos_label=1, zero_division=0)
    precision0 = precision_score(
        test_y, yhat, pos_label=0, zero_division=0)
    recall1 = recall_score(test_y, yhat, pos_label=1, zero_division=0)
    recall0 = recall_score(test_y, yhat, pos_label=0, zero_division=0)
    f1macro = f1_score(test_y, yhat, average='macro')
    f1micro = f1_score(test_y, yhat, average='micro')
    precision_recall_fscore_macro = precision_recall_fscore_support(test_y, yhat, average="macro")
    print('precision_recall_fscore_macro ' + str(precision_recall_fscore_macro))
    print('acc is '  + str(accuracy))
    print('precision1 is '  + str(precision1))
    print('precision0 is '  + str(precision0))
    print('recall1 is '  + str(recall1))
    print('recall0 is '  + str(recall0))
    print('f1macro is '  + str(f1macro))
    print('f1micro is '  + str(f1micro))

    with open(output_file, "a") as f:
        f.write(f"Generated results at {datetime.now()}\n")
        f.write("------------------------------------------\n")
        f.write(f"Modifed? {modified}\n")
        f.write(f"AUC: {auc_score}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision Recall FScore Macro: {precision_recall_fscore_macro}\n")
        f.write(f"Precision 1: {precision1}\n")
        f.write(f"Precision 0: {precision0}\n")
        f.write(f"Recall 1: {recall1}\n")
        f.write(f"Recall 0: {recall0}\n")
        f.write(f"F1 Macro: {f1macro}\n")
        f.write(f"F1 Micro: {f1micro}\n")
        f.write("\n")

    
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
    # print(train_X.head(5))
    # print(train_y.head(5))
    # print(test_X.head(5))
    # print(test_y.head(5))

    embedding_name = f"{name}_{split}_seed_{seed}_fraction_{fraction}_modified_{modified}"
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
