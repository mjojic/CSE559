import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DoubleLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout, device):
        super(DoubleLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        # Embedding matrices - 20 amino acids + padding
        self.tcr_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        # MLP
        self.hidden_layer = nn.Linear(lstm_dim * 2, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, tcrs, tcr_lens, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        tcr_embeds = self.tcr_embedding(tcrs)
        # LSTM Acceptor
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embeds, tcr_lens)
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1] for i, j in enumerate(tcr_lens)]).view(len(tcr_lens),
                                                                                                      self.lstm_dim)

        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(peps)
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens),
                                                                                                      self.lstm_dim)

        # MLP Classifier
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = F.sigmoid(mlp_output)
        return output


class ModifiedLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout, device):
        super(ModifiedLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        # Embedding matrices - 20 amino acids + padding
        self.tcr_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        # New - Cross Attention
        self.projection_q = nn.Linear(lstm_dim * 2, lstm_dim * 2)
        self.projection_k = nn.Linear(lstm_dim * 2, lstm_dim * 2)
        self.projection_v = nn.Linear(lstm_dim * 2, lstm_dim * 2)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=2 * lstm_dim, num_heads=4)
        # MLP
        self.hidden_layer = nn.Linear(lstm_dim * 2, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, tcrs, tcr_lens, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        tcr_embeds = self.tcr_embedding(tcrs)
        # LSTM Acceptor
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embeds, tcr_lens)
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1] for i, j in enumerate(tcr_lens)]).view(len(tcr_lens),
                                                                                                      self.lstm_dim)

        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(peps)
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens),
                                                                                                      self.lstm_dim)

        # New - Multihead Attention
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
        query = self.projection_q(tcr_pep_concat)
        query = query.unsqueeze(0)
        # print(f"Query shape: {query.shape}")
        key = self.projection_k(tcr_pep_concat)
        key = key.unsqueeze(0)
        value = self.projection_v(tcr_pep_concat)
        value = value.unsqueeze(0)
        attention_out, _ = self.multihead_attention(query, key, value)
        attention_out = attention_out.squeeze()
        # print(f"Attention out: {attention_out.shape}")

        # MLP Classifier
        hidden_output = self.dropout(self.relu(self.hidden_layer(attention_out)))
        mlp_output = self.output_layer(hidden_output)
        output = F.sigmoid(mlp_output)
        return output


class ProjectionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionModule, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        # self.activation = nn.SiLU()
    def forward(self, x):
        x = self.projection(x)
        x = self.dropout(x)
        # x = self.activation(x)
        x = F.silu(x)
        return x

class LSTM_ProjectionNet(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout, device, projection_dim, num_projections):
        super(LSTM_ProjectionNet, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        # Embedding matrices - 20 amino acids + padding
        self.tcr_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        # New - Cross Attention
        self.tcr_attn = nn.MultiheadAttention(embed_dim=lstm_dim, num_heads=4)
        self.pep_attn = nn.MultiheadAttention(embed_dim=lstm_dim, num_heads=4)

        self.net_pep_dim = lstm_dim
        self.net_tcr_dim = lstm_dim
        self.projection_matrices_epi = nn.ModuleList([ProjectionModule(self.net_pep_dim, projection_dim) for _ in range(num_projections)])
        self.projection_matrices_tcr = nn.ModuleList([ProjectionModule(self.net_tcr_dim, projection_dim) for _ in range(num_projections)])
        # MLP
        self.hidden_layer = nn.Linear(num_projections, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, tcrs, tcr_lens, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        tcr_embeds = self.tcr_embedding(tcrs)
        print(f"TCR Embeds: {tcr_embeds.shape}")
        # LSTM Acceptor
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embeds, tcr_lens)
        print(f"TCR LSTM out: {tcr_lstm_out.shape}")
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1] for i, j in enumerate(tcr_lens)]).view(len(tcr_lens),
                                                                                                      self.lstm_dim)
        print(f"TCR last cell: {tcr_last_cell.shape}")

        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(peps)
        print(f"Pep Embeds: {pep_embeds.shape}")
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        print(f"Pep LSTM out: {pep_lstm_out.shape}")
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens),
                                                                                                      self.lstm_dim)
        print(f"Pep last cell: {pep_last_cell.shape}")

        # New - Multihead Attention
        tcr_query = tcr_last_cell.unsqueeze(0)
        tcr_key = tcr_last_cell.unsqueeze(0)
        tcr_value = tcr_last_cell.unsqueeze(0)
        tcr_attn_out, _ = self.tcr_attn(tcr_query, tcr_key, tcr_value)
        tcr = tcr_attn_out.squeeze()

        pep_query = pep_last_cell.unsqueeze(0)
        pep_key = pep_last_cell.unsqueeze(0)
        pep_value = pep_last_cell.unsqueeze(0)
        pep_attn_out, _ = self.pep_attn(pep_query, pep_key, pep_value)
        pep = pep_attn_out.squeeze()
        

        pep = torch.transpose(pep, 0, 1)
        tcr = torch.transpose(tcr, 0, 1)
        pep = pep.reshape(-1, pep.size(-2) * pep.size(-1)) # [batch, 550]
        tcr = tcr.reshape(-1, tcr.size(-2) * tcr.size(-1)) # [batch, 500]
        print(f"Pep shape: {pep.shape}")
        print(f"TCR shape: {tcr.shape}")
        pep_features = [projection(pep) for projection in self.projection_matrices_epi] # [batch, projection_dim]
        tcr_features = [projection(tcr) for projection in self.projection_matrices_tcr] # [batch, projection_dim]

        # For each projection in pep_features, compute dot product with tcr_features, and get a list of dot products (1,1)
        dot_products = [torch.bmm(pep_feature.unsqueeze(1), tcr_feature.unsqueeze(2)).squeeze() for pep_feature, tcr_feature in zip(pep_features, tcr_features)]

        # Concatenate all dot products to form a set of features
        dot_products = torch.stack(dot_products, dim=1)

        # MLP Classifier
        hidden_output = self.dropout(self.relu(self.hidden_layer(dot_products)))
        mlp_output = self.output_layer(hidden_output)
        output = F.sigmoid(mlp_output)
        return output


class PaddingAutoencoder(nn.Module):
    def __init__(self, input_len, input_dim, encoding_dim):
        super(PaddingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.encoding_dim = encoding_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_len * self.input_dim, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, self.encoding_dim))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, self.input_len * self.input_dim))

    def forward(self, batch_size, padded_input):
        concat = padded_input.view(batch_size, self.input_len * self.input_dim)
        encoded = self.encoder(concat)
        decoded = self.decoder(encoded)
        decoding = decoded.view(batch_size, self.input_len, self.input_dim)
        decoding = F.softmax(decoding, dim=2)
        return decoding

    pass


class AutoencoderLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, device, max_len, input_dim, encoding_dim, batch_size, ae_file, train_ae):
        super(AutoencoderLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = encoding_dim
        self.max_len = max_len
        self.input_dim = input_dim
        self.batch_size = batch_size
        # TCR Autoencoder
        self.autoencoder = PaddingAutoencoder(max_len, input_dim, encoding_dim)
        checkpoint = torch.load(ae_file, map_location=device)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        if train_ae is False:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        self.autoencoder.eval()
        # Embedding matrices - 20 amino acids + padding
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.pep_lstm = nn.LSTM(embedding_dim, self.lstm_dim, num_layers=2, batch_first=True, dropout=0.1)
        # MLP
        self.mlp_dim = self.lstm_dim + encoding_dim
        self.hidden_layer = nn.Linear(self.mlp_dim, self.mlp_dim // 2)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(self.mlp_dim // 2, 1)
        self.dropout = nn.Dropout(p=0.1)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, padded_tcrs, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        concat = padded_tcrs.view(self.batch_size, self.max_len * self.input_dim)
        encoded_tcrs = self.autoencoder.encoder(concat)
        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(peps)
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens),
                                                                                                      self.lstm_dim)
        # MLP Classifier
        tcr_pep_concat = torch.cat([encoded_tcrs, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = F.sigmoid(mlp_output)
        return output
