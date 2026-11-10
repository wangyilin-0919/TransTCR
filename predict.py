import os
import argparse
import numpy as np
import pandas as pd
import math
import torch
from torch import nn
from torch.utils import data
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, auc, precision_score, recall_score, matthews_corrcoef, roc_curve


# ==========================================
# 1. 基础组件与模型定义 (保持原样)
# ==========================================
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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
    device = X.device
    mask = torch.arange((maxlen), dtype=torch.float32, device=device)[None, :].to(device) < valid_len[:, None].to(device)
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
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
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
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout_addnorm, dropout_attention, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout_attention, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout_addnorm)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout_addnorm)
        
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class Encoder(nn.Module):
    def __init__(self, key_size, query_size, value_size, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout_embedding, dropout_addnorm, dropout_attention, use_bias=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = Embeddings(vocab_size, num_hiddens, dropout_embedding, max_len=200)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout_addnorm, dropout_attention, use_bias))
            
    def forward(self, tokens, segments, valid_lens, *args):
        X = self.embedding(tokens, segments) * math.sqrt(self.num_hiddens)
        X = X.to(torch.float32)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout_addnorm, dropout_attention, use_bias=False, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout_attention, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout_addnorm)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout_addnorm)
        
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class Decoder(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout_embedding, dropout_addnorm, dropout_attention, use_bias=False, max_len=200, **kwargs):
        super(Decoder, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i), DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout_addnorm, dropout_attention, use_bias))
            
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
        self.encoder_MHC = Encoder(key_size=64, query_size=64, value_size=64, vocab_size=64, num_hiddens=64, norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128, num_heads=2, num_layers=1, dropout_embedding=0.01, dropout_addnorm=0.01, dropout_attention=0.01, use_bias=False)
        self.encoder_peptide = Encoder(key_size=64, query_size=64, value_size=64, vocab_size=64, num_hiddens=64, norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128, num_heads=2, num_layers=1, dropout_embedding=0.01, dropout_addnorm=0.01, dropout_attention=0.01, use_bias=False)
        self.pMHC_decoder = Decoder(key_size=64, query_size=64, value_size=64, vocab_size=64, num_hiddens=64, norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128, num_heads=1, num_layers=1, dropout_embedding=0.01, dropout_addnorm=0.01, dropout_attention=0.01, use_bias=False)
        self.fnc = nn.Sequential(nn.Linear(3200, 1028), nn.ReLU(), nn.BatchNorm1d(1028), nn.Linear(1028, 2))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, tokens_MHC_res, segments_MHC_res, valid_len_MHC_res, tokens_peptide_res, segments_peptide_res, valid_len_peptide_res, valid_len):
        MHC = self.encoder_MHC(tokens_MHC_res, segments_MHC_res, valid_len_MHC_res)
        peptide = self.encoder_peptide(tokens_peptide_res, segments_peptide_res, valid_len_peptide_res)
        X_encoder = torch.cat((peptide, MHC), dim=1)
        X_decoder = self.pMHC_decoder(X_encoder, valid_len)
        X = self.fnc(X_decoder.reshape(X_decoder.shape[0], -1))
        return self.softmax(X), X_encoder, X_decoder

class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens), nn.ReLU(), nn.LayerNorm(num_hiddens), nn.Linear(num_hiddens, vocab_size))
        
    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
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
        self.encoder = Encoder(key_size=64, query_size=64, value_size=64, vocab_size=24, num_hiddens=64, norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128, num_heads=2, num_layers=2, dropout_embedding=0.01, dropout_addnorm=0.01, dropout_attention=0.01, use_bias=False)
        self.hidden = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
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
        self.decoder = Decoder(key_size=64, query_size=64, value_size=64, vocab_size=64, num_hiddens=64, norm_shape=[64], ffn_num_input=64, ffn_num_hiddens=128, num_heads=1, num_layers=2, dropout_embedding=0, dropout_addnorm=0, dropout_attention=0, use_bias=False)
        self.fnc = nn.Sequential(nn.Linear(6208, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Linear(2048, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Linear(64, 2))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_TCR, encoder_pMHC, valid_len_pMT_res):
        X = torch.cat((encoder_TCR, encoder_pMHC), dim=1)
        X = self.decoder(X, valid_len_pMT_res)
        X = self.fnc(X.reshape(X.shape[0], -1))
        return self.softmax(X)

def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class TCR_DataSet(Data.Dataset):
    def __init__(self, data):
        super(TCR_DataSet, self).__init__()
        vocab = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19, "<cls>": 20, "<sep>": 21, "<pad>": 22}
        TRB = data["TCRB"].to_list()
        TRA = data["TCRA"].to_list()
        MHC = data["MHC"].to_list()
        peptide = data["Peptide"].to_list()
        labels = data["labels"].to_list() if "labels" in data.columns else [0] * len(TRB)
        
        tokens_TCR_res, segments_TCR_res, valid_len_TCR_res = [], [], []
        tokens_MHC_res, segments_MHC_res, valid_len_MHC_res = [], [], []
        tokens_peptide_res, segments_peptide_res, valid_len_peptide_res = [], [], []
        valid_len_pMHC_res, valid_len_pMT_res = [], []
        
        for i in range(len(TRB)):
            if TRA[i] == "None":
                tokens_TCR, segments_TCR = get_tokens_and_segments(list(TRB[i]), None)
            else:
                tokens_TCR, segments_TCR = get_tokens_and_segments(list(TRB[i]), list(TRA[i]))
            valid_len_TCR_res.append(len(tokens_TCR))
            tokens_TCR = tokens_TCR + ["<pad>"] * (47 - len(tokens_TCR))
            tokens_TCR_res.append([vocab[a] for a in tokens_TCR])
            segments_TCR_res.append(segments_TCR + [2] * (47 - len(segments_TCR)))
            
            tokens_MHC, segments_MHC = get_tokens_and_segments(list(MHC[i]), None)
            valid_len_MHC_res.append(len(tokens_MHC))
            tokens_MHC = tokens_MHC + ["<pad>"] * (36 - len(tokens_MHC))
            tokens_MHC_res.append([vocab[a] for a in tokens_MHC])
            segments_MHC_res.append(segments_MHC + [2] * (36 - len(segments_MHC)))
            
            tokens_peptide, segments_peptide = get_tokens_and_segments(list(peptide[i]), None)
            valid_len_peptide_res.append(len(tokens_peptide))
            tokens_peptide = tokens_peptide + ["<pad>"] * (14 - len(tokens_peptide))
            tokens_peptide_res.append([vocab[a] for a in tokens_peptide])
            segments_peptide_res.append(segments_peptide + [2] * (14 - len(segments_peptide)))
            
            valid_len_pMT_res.append(97)
            valid_len_pMHC_res.append(50)
            
        self.tokens_TCR_res = torch.tensor(tokens_TCR_res)
        self.segments_TCR_res = torch.tensor(segments_TCR_res)
        self.valid_len_TCR_res = torch.tensor(valid_len_TCR_res)
        self.tokens_MHC_res = torch.tensor(tokens_MHC_res)
        self.segments_MHC_res = torch.tensor(segments_MHC_res)
        self.valid_len_MHC_res = torch.tensor(valid_len_MHC_res)
        self.tokens_peptide_res = torch.tensor(tokens_peptide_res)
        self.segments_peptide_res = torch.tensor(segments_peptide_res)
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


# ==========================================
# 2. 推理与评估逻辑
# ==========================================
def predict_and_evaluate(args):
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    # 加载模型
    print("Loading models...")
    TCR_encoder = Transformer()
    TCR_encoder.load_state_dict(torch.load(args.tcr_model, map_location=device, weights_only=True))
    TCR_encoder = TCR_encoder.to(device)
    TCR_encoder.eval()
    
    pMHC_encoder = pMHC_net()
    pMHC_encoder.load_state_dict(torch.load(args.pmhc_model, map_location=device, weights_only=True))
    pMHC_encoder = pMHC_encoder.to(device)
    pMHC_encoder.eval()

    model = TCR_net()
    model.load_state_dict(torch.load(args.transtcr_model, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 读取数据
    print(f"Loading data from {args.input_csv}...")
    test_df = pd.read_csv(args.input_csv)
    
    # 1. 将MHC列改名叫MHC_name
    if 'MHC' in test_df.columns:
        test_df.rename(columns={'MHC': 'MHC_name'}, inplace=True)
    else:
        raise ValueError("Input CSV must contain a 'MHC' column.")

    # 2. 读取PseudoSequence_seq.csv并进行映射
    # 假设伪序列文件是用 Tab 或空格分隔的，如果没有表头，我们手动指定列名
    pseudo_df = pd.read_csv("./test_data/pMHC_pretrain_test/PseudoSequence_seq.csv", header=None, names=['HLA_name', 'sequence'])
    
    # 构建字典映射 (去除可能存在的空格)
    pseudo_df['HLA_name'] = pseudo_df['HLA_name'].str.strip()
    pseudo_df['sequence'] = pseudo_df['sequence'].str.strip()
    pseudo_df = pseudo_df[(pseudo_df['sequence'] != 'nan') & (pseudo_df['sequence'] != '')]
    hla_seq_map = dict(zip(pseudo_df['HLA_name'], pseudo_df['sequence']))

    # 3. 根据MHC_name匹配序列，替换MHC列
    test_df['MHC_name'] = test_df['MHC_name'].astype(str).str.strip()
    test_df['MHC'] = test_df['MHC_name'].map(hla_seq_map)

    # 检查是否有未匹配到的HLA
    missing_mhc = test_df[test_df['MHC'].isna()]['MHC_name'].unique()
    if len(missing_mhc) > 0:
        raise ValueError(f"Could not find pseudo-sequences for the following MHC alleles: {list(missing_mhc)}. Please check your input or the pseudo-sequence file.")

    # 后续处理（原来的 fillna 逻辑）
    test_df = test_df.fillna("None")
    has_labels = "labels" in test_df.columns
    
    batch_size = args.batch_size if args.batch_size else len(test_df["TCRB"].to_list())
    test_data_iter = Data.DataLoader(TCR_DataSet(test_df), batch_size=batch_size, shuffle=False)

    all_outputs, all_output_softmax, all_labels = [], [], []

    print("Starting prediction...")
    with torch.no_grad():
        for step, (data, label) in enumerate(test_data_iter):
            tokens_TCR_res, segments_TCR_res, valid_len_TCR_res = data[0].to(device), data[1].to(device), data[2].to(device)
            tokens_MHC_res, segments_MHC_res, valid_len_MHC_res = data[3].to(device), data[4].to(device), data[5].to(device)
            tokens_peptide_res, segments_peptide_res, valid_len_peptide_res = data[6].to(device), data[7].to(device), data[8].to(device)
            valid_len_pMT_res, valid_len_pMHC_res = data[9].to(device), data[10].to(device)
            encoder_TCR, _, _ = TCR_encoder(tokens_TCR_res, segments_TCR_res, valid_len_TCR_res, None)
            _, encoder_pMHC, decoder_pMHC = pMHC_encoder(tokens_MHC_res, segments_MHC_res, valid_len_MHC_res, 
                                                         tokens_peptide_res, segments_peptide_res, valid_len_peptide_res, 
                                                         valid_len_pMHC_res)
            output = model(encoder_TCR, encoder_pMHC, valid_len_pMT_res)
            _, predicted = torch.max(output, 1)
            all_outputs.extend(predicted.cpu().numpy())
            all_output_softmax.extend(output.cpu().numpy())
            all_labels.extend(label.numpy())

    all_outputs = np.array(all_outputs)
    all_output_softmax = np.array(all_output_softmax)
    all_labels = np.array(all_labels)

    # 写入预测结果
    test_df["Prediction"] = all_outputs
    test_df["Score"] = all_output_softmax[:, 1]
    
    # 为了保持输出文件整洁，可以将原本的MHC序列列删掉，或者保留以作核对。这里保留，方便用户查看。
    test_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransTCR Prediction Script")
    
    # 必需参数
    parser.add_argument("--input_csv", type=str, required=True, 
                        help="Path to the input CSV file containing TCRB, TCRA, MHC, Peptide, labels.")
    # 可选参数 (带默认值)
    parser.add_argument("--output_csv", type=str, default="./output/predictions.csv", 
                        help="Path to save the output CSV file with predictions.")
    parser.add_argument("--tcr_model", type=str, default="./model/TCR_pretrain_model.pkl", 
                        help="Path to the pretrained TCR model.")
    parser.add_argument("--pmhc_model", type=str, default="./model/pMHC_pretrain_model.pkl", 
                        help="Path to the pretrained pMHC model.")
    parser.add_argument("--transtcr_model", type=str, default="./model/TransTCR_model.pkl",
                        help="Path to the downstream TransTCR model.")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device to run inference ('cuda:0' or 'cpu'). Defaults to cuda if available.")
    parser.add_argument("--batch_size", type=int, default=2048, 
                        help="Batch size for inference. Adjust based on your GPU memory.")

    args = parser.parse_args()
    
    setup_seed(42)
    predict_and_evaluate(args)