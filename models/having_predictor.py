import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_name_encode, run_GAT

class HavingPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, use_syn):
        super(HavingPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu
        self.use_hs = use_hs
        self.use_syn = use_syn

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_att = nn.Linear(N_h, N_h)
        self.projection = nn.Linear(768, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.hv_out_q = nn.Linear(N_h, N_h)
        self.hv_out_hs = nn.Linear(N_h, N_h)
        self.hv_out_c = nn.Linear(N_h, N_h)
        self.multi_out_syn = nn.Linear(N_h, N_h)
        self.hv_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 2)) #for having/none

        self.softmax = nn.Softmax(dim=1)
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

        if self.use_syn:
            #initializing the linear transformation for GATs
            in_dim = N_word
            hidden_dim = 50
            out_dim = N_h
            num_heads = int(4)

            self.fc_L1 = nn.Linear(in_dim, hidden_dim, bias=False)
            self.attn_fc_L1 = nn.Linear(2 * hidden_dim, 1, bias=False)
            self.fc_L2 = nn.Linear(hidden_dim * num_heads, out_dim, bias=False)
            self.attn_fc_L2 = nn.Linear(2 * out_dim, 1, bias=False)

        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col, syn_graph, syn_feat, syn_len):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        max_syn_len = max(syn_len)
        B = len(q_len)

        # q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        q_enc = self.projection(q_emb_var)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = col_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)

        # get target/predicted column's embedding
        # col_emb: (B, hid_dim)
        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb)

        if self.use_syn:
            syn_enc = run_GAT(self.N_h, syn_graph, syn_feat, syn_len,\
                              self.fc_L1, self.attn_fc_L1, self.fc_L2, self.attn_fc_L2)

            att_syn_qmkw = torch.bmm(col_emb.unsqueeze(1), self.q_att(syn_enc).transpose(1, 2)).view(B,-1)
            for idx, num in enumerate(syn_len):
                if num < max_syn_len:
                    att_syn_qmkw[idx, num:] = -100
            att_prob_syn_qmkw  = self.softmax(att_syn_qmkw)
            syn_weighted = (syn_enc * att_prob_syn_qmkw.unsqueeze(2)).sum(1)



        att_val_qc = torch.bmm(col_emb.unsqueeze(1), self.q_att(q_enc).transpose(1, 2)).view(B,-1)
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, num:] = -100
        att_prob_qc = self.softmax(att_val_qc)
        q_weighted = (q_enc * att_prob_qc.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc = torch.bmm(col_emb.unsqueeze(1), self.hs_att(hs_enc).transpose(1, 2)).view(B,-1)
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc[idx, num:] = -100
        att_prob_hc = self.softmax(att_val_hc)
        hs_weighted = (hs_enc * att_prob_hc.unsqueeze(2)).sum(1)
        # hv_score: (B, 2)
        if self.use_syn:
            hv_score = self.hv_out(self.hv_out_q(q_weighted) + int(self.use_hs)* self.hv_out_hs(hs_weighted) + \
                                   self.multi_out_syn(syn_weighted) + self.hv_out_c(col_emb))
        else:
            hv_score = self.hv_out(self.hv_out_q(q_weighted) + int(self.use_hs)* self.hv_out_hs(hs_weighted) + self.hv_out_c(col_emb))


        return hv_score


    def loss(self, score, truth):
        loss = 0
        data = torch.from_numpy(np.array(truth))
        truth_var = Variable(data.cuda())
        loss = self.CE(score, truth_var)

        return loss


    def check_acc(self, score, truth):
        err = 0
        B = len(score)
        pred = []
        for b in range(B):
            pred.append(np.argmax(score[b].data.cpu().numpy()))
        for b, (p, t) in enumerate(zip(pred, truth)):
            if p != t:
                err += 1

        return err
