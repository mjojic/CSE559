import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import time
import numpy as np
import torch.autograd as autograd
from ERGO_models import DoubleLSTMClassifier, ModifiedLSTMClassifier, LSTM_ProjectionNet
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score, roc_curve, accuracy_score


def get_lists_from_pairs(pairs):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label, weight = pair
        tcrs.append(tcr)
        peps.append(pep)
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs


def convert_data(tcrs, peps, amino_to_ix):
    for i in range(len(tcrs)):
        if any(letter.islower() for letter in tcrs[i]):
            print(tcrs[i])
        tcrs[i] = [amino_to_ix[amino] for amino in tcrs[i]]
    for i in range(len(peps)):
        peps[i] = [amino_to_ix[amino] for amino in peps[i]]


def get_batches(tcrs, peps, signs, batch_size):
    """
    Get batches from the data
    """
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs):
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
    # Return list of all batches
    return batches


def get_full_batches(tcrs, peps, signs, batch_size, amino_to_ix):
    """
    Get batches from the data, including last with padding
    """
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs) // batch_size * batch_size:
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
    # pad data in last batch
    missing = batch_size - len(tcrs) + index
    if missing < batch_size:
        padding_tcrs = ['A'] * missing
        padding_peps = ['A' * (batch_size - missing)] * missing
        convert_data(padding_tcrs, padding_peps, amino_to_ix)
        batch_tcrs = tcrs[index:] + padding_tcrs
        batch_peps = peps[index:] + padding_peps
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        batch_signs = [0.0] * batch_size
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
        # Update index
        index += batch_size
    # Return list of all batches
    return batches
    pass


def pad_batch(seqs):
    """
    Pad a batch of sequences (part of the way to use RNN batching in PyTorch)
    """
    # Tensor of sequences lengths
    lengths = torch.LongTensor([len(seq) for seq in seqs])
    # The padding index is 0
    # Batch dimensions is number of sequences * maximum sequence length
    longest_seq = max(lengths)
    batch_size = len(seqs)
    # Pad the sequences. Start with zeros and then fill the true sequence
    padded_seqs = autograd.Variable(torch.zeros((batch_size, longest_seq))).long()
    for i, seq_len in enumerate(lengths):
        seq = seqs[i]
        padded_seqs[i, 0:seq_len] = torch.LongTensor(seq[:seq_len])
    # Return padded batch and the true lengths
    return padded_seqs, lengths


def train_epoch(batches, model, loss_function, optimizer, device):
    model.train()
    shuffle(batches)
    total_loss = 0
    total_batches = len(batches)
    current_batch = 0
    for batch in batches:
        # print(f"\tCurrently on batch {current_batch} of {total_batches-1}")
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        batch_signs = torch.tensor(batch_signs).to(device)
        model.zero_grad()
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        # print(probs, batch_signs)
        # Compute loss
        weights = batch_signs * 0.84 + (1-batch_signs) * 0.14
        loss_function.weight = weights
        loss = loss_function(probs, batch_signs)
        # with open(sys.argv[1], 'a+') as loss_file:
        #    loss_file.write(str(loss.item()) + '\n')
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print('current loss:', loss.item())
        # print(probs, batch_signs)
        current_batch += 1
    # Return average loss
    return total_loss / len(batches)


def train_model(batches, test_batches, device, args, params):
    """
    Train and evaluate the model
    """
    full_metrics_dict: dict[str, list[float]] = {}
    losses = []
    # We use Cross-Entropy loss
    loss_function = nn.BCELoss()
    # Set model with relevant parameters
    print("Making model")
    if args['siamese'] is True:
        model = SiameseLSTMClassifier(params['emb_dim'], params['lstm_dim'], device)
    elif "modified" in args and args["modified"] is True:
        print("Modified LSTM selected")
        model = ModifiedLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device)
        # model = LSTM_ProjectionNet(
        #     embedding_dim=params['emb_dim'], 
        #     lstm_dim=params['lstm_dim'], 
        #     dropout=params['dropout'], 
        #     device=device,
        #     projection_dim=params['proj_dim'],
        #     num_projections=params['num_proj'],
        # )
    else:
        print("Regular LSTM selected")
        model = DoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device)
    # Move to GPU
    print(f"Moving model to {device}")
    model.to(device)
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    # Train several epochs
    # best_auc = 0
    # best_roc = None
    print("Beginning training")
    for epoch in range(params['epochs']):
        print('epoch:', epoch + 1)
        epoch_time = time.time()
        # Train model and get loss
        loss = train_epoch(batches, model, loss_function, optimizer, device)
        losses.append(loss)
        # Get the metrics for this epoch
        metrics_dict = get_metrics_dict(model=model, batches=test_batches, device=device)
        for metric, value in metrics_dict.items():
            if not (metric in full_metrics_dict) or full_metrics_dict[metric] is None:
                full_metrics_dict[metric] = []
            full_metrics_dict[metric].append(value)
        print('one epoch time:', time.time() - epoch_time)
    full_metrics_dict["loss"] = losses
    # return model, best_auc, best_roc
    return model, full_metrics_dict


def get_metrics_dict(model, batches, device):
    model.eval()
    true = []
    scores = []
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
    # Return metrics
    metrics_dict = {}
    metrics_dict['auc_score'] = roc_auc_score(true, scores)
    
    # scores[scores>=0.5] = 1
    # scores[scores<0.5] = 0
    scores = list(map(lambda score: 1.0 if score >= 0.5 else 0.0, scores))
    
    metrics_dict['accuracy'] = accuracy_score(true, scores)
    metrics_dict['precision1'] = precision_score(
        true, scores, pos_label=1, zero_division=0)
    metrics_dict['precision0'] = precision_score(
        true, scores, pos_label=0, zero_division=0)
    metrics_dict['recall1'] = recall_score(true, scores, pos_label=1, zero_division=0)
    metrics_dict['recall0'] = recall_score(true, scores, pos_label=0, zero_division=0)
    metrics_dict['f1macro'] = f1_score(true, scores, average='macro')
    metrics_dict['f1micro'] = f1_score(true, scores, average='micro')
    metrics_dict['precision_recall_fscore_macro'] = precision_recall_fscore_support(true, scores, average="macro")
    return metrics_dict


def evaluate(model, batches, device):
    model.eval()
    true = []
    scores = []
    shuffle(batches)
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        # print(np.array(batch_signs).astype(int))
        # print(probs.cpu().data.numpy())
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
    # Return auc score
    auc = roc_auc_score(true, scores)
    fpr, tpr, thresholds = roc_curve(true, scores)
    return auc, (fpr, tpr, thresholds)


def evaluate_full(model, batches, device):
    model.eval()
    true = []
    scores = []
    index = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
        batch_size = len(tcr_lens)
        assert batch_size == 50
        index += batch_size
    border = pep_lens[-1]
    if any(k != border for k in pep_lens[border:]):
        print(pep_lens)
    else:
        index -= batch_size - border
        true = true[:index]
        scores = scores[:index]
    if int(sum(true)) == len(true) or int(sum(true)) == 0:
        # print(true)
        raise ValueError
    # Return auc score
    auc = roc_auc_score(true, scores)
    fpr, tpr, thresholds = roc_curve(true, scores)
    return auc, (fpr, tpr, thresholds)


def predict(model, batches, device):
    model.eval()
    preds = []
    index = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        preds.extend([t[0] for t in probs.cpu().data.tolist()])
        batch_size = len(tcr_lens)
        assert batch_size == 50
        index += batch_size
    border = pep_lens[-1]
    if any(k != border for k in pep_lens[border:]):
        print(pep_lens)
    else:
        index -= batch_size - border
        preds = preds[:index]
    return preds
