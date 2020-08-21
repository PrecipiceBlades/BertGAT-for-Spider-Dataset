import json
import torch
import datetime
import argparse
import numpy as np
from utils import *
from supermodel import SuperModel
import pickle
from allennlp.predictors.predictor import Predictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')
    parser.add_argument('--toy', action='store_true',
                        help='If set, use small data; used for fast debugging.')
    parser.add_argument('--models', type=str, help='path to saved model')
    parser.add_argument('--test_data_path',type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--history_type', type=str, default='full', choices=['full','part','no'], help='full, part, or no history')
    parser.add_argument('--table_type', type=str, default='std', choices=['std','hier','no'], help='standard, hierarchical, or no table info')
    parser.add_argument('--use_syntax', type=str, default='True', choices=['False','True'], help='Use GAT to extract syntax information')
    parser.add_argument('--pre_syntax', type=str, default='False', choices=['False','True'], help='preprocessing syntax information')
    args = parser.parse_args()
    use_hs = True
    pre_syntax = False

    if args.history_type == "no":
        args.history_type = "full"
        use_hs = False

    if args.use_syntax == 'True':
        print ('Using syntax infomration')
        use_syn = True
    else:
        print ('Not using syntax information')
        use_syn = False

    if args.pre_syntax == 'True':
        pre_syntax = True

    N_word=300
    B_word=42
    N_h = 300
    N_depth=2
    # if args.part:
    #     part = True
    # else:
    #     part = False
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=2 #20
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=2 #64
    # TRAIN_ENTRY=(False, True, False)  # (AGG, SEL, COND)
    # TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-4

    #TODO remove the toy setting
    USE_SMALL=True

    #TODO
    print ('Loading the data')
    data = json.load(open(args.test_data_path))
    # dev_data = load_train_dev_dataset(args.train_component, "dev", args.history)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=args.train_emb, use_small=USE_SMALL)
    # dev_data = load_train_dev_dataset(args.train_component, "dev", args.history)
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")

    if pre_syntax == True:
        print ('Start preprocessing the syntax information')
        preprocess_syntax_test(data,args.test_data_path)
        print ('Syntaxt preprocessing is done')
        exit()


    if use_syn:
        syn_data = pickle.load(open(args.test_data_path+'.p','rb'))




    model = SuperModel(word_emb, N_word=N_word, gpu=GPU, trainable_emb = args.train_emb, table_type=args.table_type, use_hs=use_hs, use_syn=use_syn)

    # agg_m, sel_m, cond_m = best_model_name(args)
    # torch.save(model.state_dict(), "saved_models/{}_models.dump".format(args.train_component))


    print ("Loading from modules...")
    model.multi_sql.load_state_dict(torch.load("{}/multi_sql_models.dump".format(args.models)))
    model.key_word.load_state_dict(torch.load("{}/keyword_models.dump".format(args.models)))
    model.col.load_state_dict(torch.load("{}/col_models.dump".format(args.models)))
    model.op.load_state_dict(torch.load("{}/op_models.dump".format(args.models)))
    model.agg.load_state_dict(torch.load("{}/agg_models.dump".format(args.models)))
    model.root_teminal.load_state_dict(torch.load("{}/root_tem_models.dump".format(args.models)))
    model.des_asc.load_state_dict(torch.load("{}/des_asc_models.dump".format(args.models)))
    model.having.load_state_dict(torch.load("{}/having_models.dump".format(args.models)))
    print("start loading_questions")
    pickle_in = open("sb2.p","rb")
    qs = pickle.load(pickle_in)
    pickle_in = open("sb.p","rb")
    decoded = pickle.load(pickle_in).get("qs")
    test_acc(model, BATCH_SIZE, data, args.output_path, qs, decoded, syn_data)
    #test_exec_acc()
