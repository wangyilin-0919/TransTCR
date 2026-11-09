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
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        self.segment_embedding = nn.Embedding(3, num_hiddens)
        
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


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        return 10000 
    distance = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            distance += 1
    return distance


def _get_next_sentence(sentence, next_sentence, paragraphs):
    if (random.random() < 0.5) or (next_sentence == None):
        is_next = True
    else:
        next_sentence_choice = random.choice(paragraphs)
        if hamming_distance(next_sentence, next_sentence_choice) < 3:
            is_next = True
            next_sentence = next_sentence_choice
        else:
            is_next = False
            next_sentence = next_sentence_choice
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(token_trb, token_tra, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    tokens_a, tokens_b, is_next = _get_next_sentence(token_trb, token_tra, paragraphs)
    if tokens_b != None:
        tokens, segments = get_tokens_and_segments(list(tokens_a), list(tokens_b))
    else:
        tokens, segments = get_tokens_and_segments(list(tokens_a), None)
    nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = "<mask>"
        # 20% of the time: keep the word unchanged
        else:
            masked_token = tokens[mlm_pred_position]
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return [vocab[a] for a in mlm_input_tokens], pred_positions, [vocab[a] for a in mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)),
                                          dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)),
                                         dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), 
                                               dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)),
                                            dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
                                           dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)


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
        self.vocab = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
                      "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
                      "<cls>": 20, "<sep>": 21, "<pad>": 22, "<mask>":23}
        TRB = data["TCRB"].to_list()
        TRA = data["TCRA"].to_list()
        TRA_list = list(set(TRA))
        for i in range(len(TRA)):
            if TRA[i] == "None":
                TRA[i] = None
        examples = []
        for i in range(len(TRB)):
            examples.extend(_get_nsp_data_from_paragraph(TRB[i], TRA[i], TRA_list, self.vocab, max_len=47))
        examples = [(_get_mlm_data_from_tokens(tokens, 
                                               self.vocab) + (segments, is_next)) for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens, 
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = _pad_bert_inputs(examples, 47, self.vocab)
        
    def __len__(self):
        return len(self.all_token_ids)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], 
                 self.all_pred_positions[idx], self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                 self.nsp_labels[idx])
                 

def train_model(model, traindataloader, criterion, optimizer, lr_scheduler, num_epochs):
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * 0.9)
    train_loss_all = []
    train_acc_mlm = []
    train_acc_nsp = []
    val_loss_all = []
    val_acc_mlm = []
    val_acc_nsp = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        train_loss = 0.0
        train_num = 0
        train_corrects_mlm = 0
        train_corrects_nsp = 0
        val_loss = 0.0
        val_num = 0
        val_corrects_mlm = 0
        val_corrects_nsp = 0
        for step, (data) in enumerate(traindataloader):
            token = data[0].to(device)
            segment = data[1].to(device)
            valid_len = data[2].to(device)
            pred_position = data[3].to(device)
            mlm_weight = data[4].to(device)
            mlm_label = data[5].to(device)
            nsp_label = data[6].to(device)
            if step < (train_batch_num-1):
                model.train()
                # prediction
                encoded_X, mlm_Y_hat, nsp_Y_hat = model(token.to(device), 
                                                        segment.to(device), 
                                                        valid_len.to(device), 
                                                        pred_position.to(device))
                # Compute masked language model loss
                mlm_l = criterion(mlm_Y_hat.reshape(-1, vocab_size), mlm_label.reshape(-1)) * mlm_weight.reshape(-1, 1)
                mlm_l = mlm_l.sum() / (mlm_weight.sum() + 1e-8)
                # Compute next sentence prediction loss
                nsp_l = criterion(nsp_Y_hat, nsp_label)
                l = mlm_l + nsp_l
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_loss += l.item() * token.size(0)
                train_num += token.size(0)
                train_corrects_mlm += torch.sum(mlm_label.reshape(-1) == torch.argmax(mlm_Y_hat, 2).reshape(-1))
                train_corrects_nsp += torch.sum(nsp_label.reshape(-1) == torch.argmax(nsp_Y_hat, 1).reshape(-1))
                torch.cuda.empty_cache()
            else:
                model.eval()
                # prediction
                encoded_X, mlm_Y_hat, nsp_Y_hat = model(token.to(device), 
                                                        segment.to(device), 
                                                        valid_len.to(device), 
                                                        pred_position.to(device))
                # Compute masked language model loss
                mlm_l = criterion(mlm_Y_hat.reshape(-1, vocab_size), mlm_label.reshape(-1)) * mlm_weight.reshape(-1, 1)
                mlm_l = mlm_l.sum() / (mlm_weight.sum() + 1e-8)
                # Compute next sentence prediction loss
                nsp_l = criterion(nsp_Y_hat, nsp_label)
                l = mlm_l + nsp_l
                val_loss += l.item() * token.size(0)
                val_num += token.size(0)
                val_corrects_mlm += torch.sum(mlm_label.reshape(-1) == torch.argmax(mlm_Y_hat, 2).reshape(-1))
                val_corrects_nsp += torch.sum(nsp_label.reshape(-1) == torch.argmax(nsp_Y_hat, 1).reshape(-1))
                torch.cuda.empty_cache()
                
        lr_scheduler.step()
        
        train_loss_all.append(train_loss / train_num)
        train_acc_mlm.append(train_corrects_mlm.double().item() / train_num)
        train_acc_nsp.append(train_corrects_nsp.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_mlm.append(val_corrects_mlm.double().item() / val_num)
        val_acc_nsp.append(val_corrects_nsp.double().item() / val_num)

        print('{} Train loss: {:4f}, train acc mlm: {:4f}, train acc nsp: {:4f}'.format(epoch, 
                                                                                        train_loss_all[-1], 
                                                                                        train_acc_mlm[-1]/7, train_acc_nsp[-1]))
        print('{} Val loss: {:4f}, val acc mlm: {:4f}, val acc nsp: {:4f}'.format(epoch, 
                                                                                  val_loss_all[-1], 
                                                                                  val_acc_mlm[-1]/7, val_acc_nsp[-1]))
        torch.cuda.empty_cache()    
        torch.cuda.empty_cache()
        time_use = time.time() - since
        print("Train complete in {:0f}m {:0f}s".format(time_use // 60, time_use % 60))
    return model


model = Transformer()
print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data = pd.read_csv("TCR_pretrain_data.csv")
train_data_iter = Data.DataLoader(TCR_DataSet(data), batch_size = 64, shuffle = True)

print("-----------------------------------------------------------------")
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss().to(device)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
model = model.to(device)
vocab_size = 24
model = train_model(model, train_data_iter, criterion, optimizer, StepLR, num_epochs=100)

torch.save(model.state_dict(), "../model/TCR_pretrain_model.pkl")
print('Model is saved')
