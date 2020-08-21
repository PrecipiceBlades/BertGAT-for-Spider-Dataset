import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from .GATNet import GAT

def run_lstm(lstm, inp, inp_len, hidden=None):
    # Run the LSTM using packed sequence.
    # This requires to first sort the input according to its length.
    sort_perm = np.array(sorted(range(len(inp_len)),
        key=lambda k:inp_len[k], reverse=True))
    sort_inp_len = inp_len[sort_perm]
    sort_perm_inv = np.argsort(sort_perm)
    if inp.is_cuda:
        sort_perm = torch.LongTensor(sort_perm).cuda()
        sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()

    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm],
            sort_inp_len, batch_first=True)
    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
    ret_s = nn.utils.rnn.pad_packed_sequence(
            sort_ret_s, batch_first=True)[0][sort_perm_inv]
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h

def run_GAT(N_h, syn_graph, syn_feat, syn_len, fc_L1, attn_fc_L1, fc_L2, attn_fc_L2):
    ret_h = torch.zeros(syn_feat.shape[0], syn_feat.shape[1], syn_feat.shape[2])
    #print('run_GAT: syn_feat.is_cuda', syn_feat.is_cuda)
    if syn_feat.is_cuda:
        ret_h = ret_h.cuda()
    for index, g in enumerate(syn_graph):
        #print ('run_GAT: index',index)
        net = GAT(g,
         fc_L1, attn_fc_L1, fc_L2, attn_fc_L2,
          num_heads=int(4))
        if syn_feat.is_cuda:
            net = net.cuda()
        logits = net(syn_feat[index][:syn_len[index]])
        #print (logits.shape)
        for i in range(logits.shape[0]):
            ret_h[index, i, :]  = logits[i,:]
    #print(ret_h, ret_h.shape)
    #input("Press Enter to continue...")
    return ret_h



def col_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    #Encode the columns.
    #The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1]
    ret = torch.FloatTensor(
            len(col_len), max(col_len), name_out.size()[1]).zero_()
    if name_out.is_cuda:
        ret = ret.cuda()

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st+cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len
