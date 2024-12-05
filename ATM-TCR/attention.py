import torch
import torch.nn as nn
import math

class Net(nn.Module):
    def __init__(self, embedding, args):
        super(Net, self).__init__()

        # Embedding Layer
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino-1)
        if not (args.blosum is None or args.blosum.lower() == 'none'):
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

        self.attn_tcr = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads)
        self.attn_pep = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads)

        # Dense Layer
        self.size_hidden1_dense = 2 * args.lin_size
        self.size_hidden2_dense = 1 * args.lin_size
        self.net_pep_dim = args.max_len_pep * self.embedding_dim
        self.net_tcr_dim = args.max_len_tcr * self.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(self.net_pep_dim + self.net_tcr_dim,
                      self.size_hidden1_dense),
            nn.BatchNorm1d(self.size_hidden1_dense),
            nn.Dropout(args.drop_rate * 2),
            nn.SiLU(),
            # nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
            # nn.BatchNorm1d(self.size_hidden2_dense),
            # nn.Dropout(args.drop_rate),
            # nn.SiLU(),
            nn.Linear(self.size_hidden2_dense, 1),
            nn.Sigmoid()
        )

    def forward(self, pep, tcr):

        # Embedding
        pep = self.embedding(pep) # batch * len * dim (25)
        tcr = self.embedding(tcr) # batch * len * dim

        pep = torch.transpose(pep, 0, 1)
        tcr = torch.transpose(tcr, 0, 1)

        # Attention
        pep, pep_attn = self.attn_pep(pep,pep,pep)
        tcr, tcr_attn = self.attn_tcr(tcr,tcr,tcr)

        pep = torch.transpose(pep, 0, 1)
        tcr = torch.transpose(tcr, 0, 1)

        # Linear
        pep = pep.reshape(-1, 1, pep.size(-2) * pep.size(-1))
        tcr = tcr.reshape(-1, 1, tcr.size(-2) * tcr.size(-1))
        peptcr = torch.cat((pep, tcr), -1).squeeze(-2)
        peptcr = self.net(peptcr)

        return peptcr

class PositionWiseEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.pos_embedding = nn.Embedding(self.max_len, embedding_dim)

    def forward(self, x):
        # inputs = [seq_len, batch_size, dim]
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        pos = torch.arange(0, seq_len).unsqueeze(1).repeat(1, batch_size).to(x.device)
        pos_embedding = self.pos_embedding(pos)
        return pos_embedding

class CrossAttentionNet(nn.Module):
    def __init__(self, embedding, args):
        super(CrossAttentionNet, self).__init__()
        # Embedding Layer
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino-1)
        self.size_hidden1_dense = 2 * args.lin_size
        self.size_hidden2_dense = 1 * args.lin_size
        self.net_pep_dim = args.max_len_pep * self.embedding_dim
        self.net_tcr_dim = args.max_len_tcr * self.embedding_dim
        if not (args.blosum is None or args.blosum.lower() == 'none'):
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

        self.attn_tcr = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads)
        self.attn_pep = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads)

        # self.pos_encoding_pep = nn.Embedding(args.max_len_pep, self.embedding_dim)
        # self.pos_encoding_tcr = nn.Embedding(args.max_len_tcr, self.embedding_dim)
        self.pos_encoding_pep = PositionWiseEmbedding(args.max_len_pep, self.embedding_dim)
        self.pos_encoding_tcr = PositionWiseEmbedding(args.max_len_tcr, self.embedding_dim)
        self.type_encoding = nn.Embedding(2, self.embedding_dim)

        self.query_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads)

        # Dense Layer
        self.net = nn.Sequential(
            nn.Linear(self.net_pep_dim + self.net_tcr_dim,
                      self.size_hidden1_dense),
            nn.BatchNorm1d(self.size_hidden1_dense),
            nn.Dropout(args.drop_rate * 2),
            nn.SiLU(),
            nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
            nn.BatchNorm1d(self.size_hidden2_dense),
            nn.Dropout(args.drop_rate),
            nn.SiLU(),
            nn.Linear(self.size_hidden2_dense, self.size_hidden2_dense),
            nn.BatchNorm1d(self.size_hidden2_dense),
            nn.Dropout(args.drop_rate),
            nn.SiLU(),
            nn.Linear(self.size_hidden2_dense, 1),
            nn.Sigmoid()
        )


    def forward(self, pep, tcr):

        # Embedding
        pep = self.embedding(pep) # batch * len * dim (25)
        tcr = self.embedding(tcr) # batch * len * dim

        pep = torch.transpose(pep, 0, 1)
        tcr = torch.transpose(tcr, 0, 1)

        # Positional Encoding
        # pep_pos_indices = torch.arange(pep.size(0), device=pep.device)
        # tcr_pos_indices = torch.arange(tcr.size(0), device=tcr.device)
        # pep_pos = self.pos_encoding_pep(pep_pos_indices).unsqueeze(1).repeat(1, pep.size(1), 1)
        # tcr_pos = self.pos_encoding_tcr(tcr_pos_indices).unsqueeze(1).repeat(1, tcr.size(1), 1)
        pep_pos = self.pos_encoding_pep(pep)
        tcr_pos = self.pos_encoding_tcr(tcr)

        # Type Encoding
        # pep_type = self.type_encoding(torch.zeros(1, dtype=torch.long, device=pep.device)).unsqueeze(0).repeat(pep.size(0), pep.size(1), 1)
        # tcr_type = self.type_encoding(torch.ones(1, dtype=torch.long, device=tcr.device)).unsqueeze(0).repeat(tcr.size(0), tcr.size(1), 1)
        
        # pep += pep_type
        # tcr += tcr_type
        pep += pep_pos
        tcr += tcr_pos

        # Attention
        pep, pep_attn = self.attn_pep(pep,pep,pep)
        tcr, tcr_attn = self.attn_tcr(tcr,tcr,tcr)

        pep, pep_attn = self.attn_pep(pep,pep,pep)
        tcr, tcr_attn = self.attn_tcr(tcr,tcr,tcr)

        
        # Cross Attention
        # cat_pep_tcr = torch.cat((pep, tcr), 0)
        # query = self.query_projection(cat_pep_tcr)
        # key = self.key_projection(cat_pep_tcr)
        # value = self.value_projection(cat_pep_tcr)

        # cross_attn, cross_attn_wei = self.cross_attention(query, key, value)
        # cross_attn = cross_attn.squeeze()

        # # Linear
        # cross_attn = cross_attn.transpose(0, 1)
        # cross_attn = cross_attn.reshape(cross_attn.size(0), -1)
        # out = self.net(cross_attn)

        pep = torch.transpose(pep, 0, 1)
        tcr = torch.transpose(tcr, 0, 1)

        # Linear
        pep = pep.reshape(-1, 1, pep.size(-2) * pep.size(-1))
        tcr = tcr.reshape(-1, 1, tcr.size(-2) * tcr.size(-1))
        peptcr = torch.cat((pep, tcr), -1).squeeze(-2)
        peptcr = self.net(peptcr)


        return peptcr