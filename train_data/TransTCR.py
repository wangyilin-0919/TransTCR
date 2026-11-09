import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import math
import random
import torch
from torch import nn
from torch.utils import data
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, auc, precision_score, recall_score, matthews_corrcoef, roc_curve, auc

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

class Embeddings(nn.Module):
    def __init__(self, vocab_size, num_hiddens, dropout, max_len):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(3, num_hiddens)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        
    def forward(self, tokens, segments):
        X = self.token_embedding(tokens) + self.segment_embedding(segments) + self.pos_embedding.data[:, :tokens.shape[1], :]
        return X


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :].to(device) < valid_len[:, None].to(device)
    X[~mask] = value
    return X
    

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        
    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.LeakyReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, 
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout_addnorm, dropout_attention, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, 
                                            num_hiddens, num_heads, 
                                            dropout_attention, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout_addnorm)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout_addnorm)
        
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class Encoder(nn.Module):
    def __init__(self, 
                 key_size, query_size, value_size, vocab_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, 
                 dropout_embedding, dropout_addnorm, dropout_attention, 
                 use_bias=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = Embeddings(vocab_size, num_hiddens, dropout_embedding, max_len=200)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, 
                             dropout_addnorm, dropout_attention, use_bias))
            
    def forward(self, tokens, segments, valid_lens, *args):
        X = self.embedding(tokens, segments) * math.sqrt(self.num_hiddens)
        X = X.to(torch.float32)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, 
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout_addnorm, dropout_attention, use_bias=False, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, 
                                            num_hiddens, num_heads, 
                                            dropout_attention, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout_addnorm)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout_addnorm)
        
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class Decoder(nn.Module):
    def __init__(self, 
                 key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, 
                 dropout_embedding, dropout_addnorm, dropout_attention, 
                 use_bias=False, max_len=200, **kwargs):
        super(Decoder, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, 
                             dropout_addnorm, dropout_attention, use_bias))
            
    def forward(self, X, valid_lens):
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        X = X.to(torch.float32)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class pMHC_net(nn.Module):
    def __init__(self, **kwargs):
        super(pMHC_net, self).__init__()
        # Encoder
        self.encoder_MHC = Encoder(key_size=64, query_size=64, value_size=64, vocab_size=64, num_hiddens=64,
                                   norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128,
                                   num_heads=2, num_layers=1, 
                                   dropout_embedding=0.01, dropout_addnorm=0.01, dropout_attention=0.01, 
                                   use_bias=False)
        self.encoder_peptide = Encoder(key_size=64, query_size=64, value_size=64, vocab_size=64, num_hiddens=64,
                                       norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128,
                                       num_heads=2, num_layers=1, 
                                       dropout_embedding=0.01, dropout_addnorm=0.01, dropout_attention=0.01, 
                                       use_bias=False)
        self.pMHC_decoder = Decoder(key_size=64, query_size=64, value_size=64, vocab_size=64, num_hiddens=64,
                                    norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128,
                                    num_heads=1, num_layers=1, 
                                    dropout_embedding=0.01, dropout_addnorm=0.01, dropout_attention=0.01, 
                                    use_bias=False)
        self.fnc = nn.Sequential(nn.Linear(3200, 1028),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(1028),
                                 nn.Linear(1028, 2))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, 
                tokens_MHC_res, segments_MHC_res, valid_len_MHC_res,
                tokens_peptide_res, segments_peptide_res, valid_len_peptide_res,
                valid_len):
        MHC = self.encoder_MHC(tokens_MHC_res, segments_MHC_res, valid_len_MHC_res)
        peptide = self.encoder_peptide(tokens_peptide_res, segments_peptide_res, valid_len_peptide_res)
        X_encoder = torch.cat((peptide, MHC), dim=1)
        X_decoder = self.pMHC_decoder(X_encoder, valid_len)
        X = self.fnc(X_decoder.reshape(X_decoder.shape[0], -1))
        return self.softmax(X), X_encoder, X_decoder


class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, num_inputs, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(), 
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))
        
    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
    
    
class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)
    

class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        # Encoder
        self.encoder = Encoder(key_size=64, query_size=64, value_size=64, vocab_size=24, num_hiddens=64,
                               norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128,
                               num_heads=2, num_layers=2, 
                               dropout_embedding=0.01, dropout_addnorm=0.01, dropout_attention=0.01, 
                               use_bias=False)
        self.hidden = nn.Sequential(nn.Linear(64, 64),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size=24, num_hiddens=64, num_inputs=64)
        self.nsp = NextSentencePred(num_inputs=64)
        
    def forward(self, tokens, segments, valid_len, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_len)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


class TCR_net(nn.Module):
    def __init__(self, **kwargs):
        super(TCR_net, self).__init__()
        # Decoder
        self.decoder = Decoder(key_size=64, query_size=64, value_size=64, vocab_size=64, num_hiddens=64,
                               norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128,
                               num_heads=1, num_layers=2, 
                               dropout_embedding=0, dropout_addnorm=0, dropout_attention=0, 
                               use_bias=False)
        # Classify Head
        self.fnc = nn.Sequential(nn.Linear(6208, 2048),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(2048),
                                 #nn.Dropout(p=0.1),
                                 nn.Linear(2048, 64),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(64),
                                 nn.Linear(64, 2))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_TCR, encoder_pMHC, valid_len_pMT_res):
        # TCR-pMHC
        X = torch.cat((encoder_TCR, encoder_pMHC), dim=1)
        # Decoder
        X = self.decoder(X, valid_len_pMT_res)
        # Classify Head
        print(X.reshape(X.shape[0], -1).shape)
        X = self.fnc(X.reshape(X.shape[0], -1))
        return self.softmax(X)


def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class TCR_DataSet(Data.Dataset):
    def __init__(self, data):
        super(TCR_DataSet, self).__init__()
        vocab = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
                 "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
                 "<cls>": 20, "<sep>": 21, "<pad>": 22}
        TRB = data["TCRB"].to_list()
        TRA = data["TCRA"].to_list()
        MHC = data["MHC"].to_list()
        peptide = data["Peptide"].to_list()
        labels = data["labels"].to_list()
        tokens_TCR_res = []
        segments_TCR_res = []
        valid_len_TCR_res = []
        tokens_MHC_res = []
        segments_MHC_res = []
        valid_len_MHC_res = []
        tokens_peptide_res = []
        segments_peptide_res = []
        valid_len_peptide_res = []
        valid_len_pMHC_res = []
        valid_len_pMT_res = []
        for i in range(len(TRB)):
            # TCR
            if TRA[i] == "None":
                tokens_TCR, segments_TCR = get_tokens_and_segments(list(TRB[i]), None)
            else:
                tokens_TCR, segments_TCR = get_tokens_and_segments(list(TRB[i]), list(TRA[i]))
            valid_len_TCR_res.append(len(tokens_TCR))
            tokens_TCR = tokens_TCR + ["<pad>"] * (47 - len(tokens_TCR))
            tokens_TCR = [vocab[a] for a in tokens_TCR]
            segments_TCR = segments_TCR + [2] * (47 - len(segments_TCR))
            tokens_TCR_res.append(tokens_TCR)
            segments_TCR_res.append(segments_TCR)
            # MHC
            tokens_MHC, segments_MHC = get_tokens_and_segments(list(MHC[i]), None)
            valid_len_MHC_res.append(len(tokens_MHC))
            tokens_MHC = tokens_MHC + ["<pad>"] * (36 - len(tokens_MHC))
            tokens_MHC = [vocab[a] for a in tokens_MHC]
            tokens_MHC_res.append(tokens_MHC)
            segments_MHC = segments_MHC + [2] * (36 - len(segments_MHC))
            segments_MHC_res.append(segments_MHC)
            # peptide
            tokens_peptide, segments_peptide = get_tokens_and_segments(list(peptide[i]), None)
            valid_len_peptide_res.append(len(tokens_peptide))
            tokens_peptide = tokens_peptide + ["<pad>"] * (14 - len(tokens_peptide))
            tokens_peptide = [vocab[a] for a in tokens_peptide]
            tokens_peptide_res.append(tokens_peptide)
            segments_peptide = segments_peptide + [2] * (14 - len(segments_peptide))
            segments_peptide_res.append(segments_peptide)
            # decoder
            valid_len_pMT_res.append(97)
            valid_len_pMHC_res.append(50)
        self.tokens_TCR_res = torch.tensor([tokens_TCR_res]).reshape(-1, 47)
        self.segments_TCR_res = torch.tensor([segments_TCR_res]).reshape(-1, 47)
        self.valid_len_TCR_res = torch.tensor(valid_len_TCR_res)
        self.tokens_MHC_res = torch.tensor([tokens_MHC_res]).reshape(-1, 36)
        self.segments_MHC_res = torch.tensor([segments_MHC_res]).reshape(-1, 36)
        self.valid_len_MHC_res = torch.tensor(valid_len_MHC_res)
        self.tokens_peptide_res = torch.tensor([tokens_peptide_res]).reshape(-1, 14)
        self.segments_peptide_res = torch.tensor([segments_peptide_res]).reshape(-1, 14)
        self.valid_len_peptide_res = torch.tensor(valid_len_peptide_res)
        self.valid_len_pMT_res = torch.tensor(valid_len_pMT_res)
        self.valid_len_pMHC_res = torch.tensor(valid_len_pMHC_res)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return self.tokens_TCR_res.shape[0]

    def __getitem__(self, idx):
        return [self.tokens_TCR_res[idx], self.segments_TCR_res[idx], self.valid_len_TCR_res[idx], 
                 self.tokens_MHC_res[idx], self.segments_MHC_res[idx], self.valid_len_MHC_res[idx],
                 self.tokens_peptide_res[idx], self.segments_peptide_res[idx], self.valid_len_peptide_res[idx],
                 self.valid_len_pMT_res[idx], self.valid_len_pMHC_res[idx]], self.labels[idx]


def train_model(model, traindataloader, train_rate, criterion, optimizer, lr_scheduler, num_epochs):
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        for step, (data, label) in enumerate(traindataloader):
            tokens_TCR_res = data[0]
            segments_TCR_res = data[1]
            valid_len_TCR_res = data[2]
            tokens_MHC_res = data[3]
            segments_MHC_res = data[4]
            valid_len_MHC_res = data[5]
            tokens_peptide_res = data[6]
            segments_peptide_res = data[7]
            valid_len_peptide_res = data[8]
            valid_len_pMT_res = data[9]
            valid_len_pMHC_res = data[10]
            labels = label
            if step < (train_batch_num-1):
                model.train()
                encoder_TCR, _, _ = TCR_encoder(tokens_TCR_res.to(device), segments_TCR_res.to(device), valid_len_TCR_res.to(device), None)
                _, encoder_pMHC, decoder_pMHC = pMHC_encoder(tokens_MHC_res.to(device), segments_MHC_res.to(device), valid_len_MHC_res.to(device),
                                                             tokens_peptide_res.to(device), segments_peptide_res.to(device), valid_len_peptide_res.to(device),
                                                             valid_len_pMHC_res.to(device))
                output = model(encoder_TCR.to(device), encoder_pMHC.to(device), valid_len_pMT_res.to(device))
                labels = labels.to(device)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, labels.long()).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * tokens_TCR_res.size(0)
                train_corrects += torch.sum(pre_lab == labels.data)
                train_num += tokens_TCR_res.size(0)
                torch.cuda.empty_cache()
            else:
                model.eval()
                encoder_TCR, _, _ = TCR_encoder(tokens_TCR_res.to(device), segments_TCR_res.to(device), valid_len_TCR_res.to(device), None)
                _, encoder_pMHC, decoder_pMHC = pMHC_encoder(tokens_MHC_res.to(device), segments_MHC_res.to(device), valid_len_MHC_res.to(device),
                                                             tokens_peptide_res.to(device), segments_peptide_res.to(device), valid_len_peptide_res.to(device),
                                                             valid_len_pMHC_res.to(device))
                output = model(encoder_TCR.to(device), encoder_pMHC.to(device), valid_len_pMT_res.to(device))
                labels = labels.to(device)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, labels.long())
                val_loss += loss.item() * tokens_TCR_res.size(0)
                val_corrects += torch.sum(pre_lab == labels.data)
                val_num += tokens_TCR_res.size(0)
                torch.cuda.empty_cache()
        lr_scheduler.step()

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train loss: {:4f} train acc: {:4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val loss: {:4f} val acc: {:4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
        torch.cuda.empty_cache()    
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:0f}m {:0f}s".format(time_use // 60, time_use % 60))
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})
    return model, train_process


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

TCR_encoder = Transformer()
TCR_encoder.load_state_dict(torch.load("./TCR_pretrain_model.pkl", map_location=torch.device('cpu')))
TCR_encoder = TCR_encoder.to(device)

pMHC_encoder = pMHC_net()
pMHC_encoder.load_state_dict(torch.load("./pMHC_pretrain_model.pkl", map_location=torch.device('cpu')))
pMHC_encoder = pMHC_encoder.to(device)

model = TCR_net()
print(model)


data = pd.read_csv("./TransTCR_train_data.csv")
print("There are " + str(len(data["TCRB"].to_list())) + " sequences in the dataset.")
train_data_iter = Data.DataLoader(TCR_DataSet(data), batch_size = 64, shuffle = True)


print("-----------------------------------------------------------------")
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss().to(device)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
model = model.to(device)
model, train_process = train_model(model, train_data_iter, 0.9, criterion, optimizer, StepLR, num_epochs=100)


torch.save(model.state_dict(), "../model/TransTCR_model.pkl")
print('Model is saved')
