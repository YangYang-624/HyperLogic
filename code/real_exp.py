import sys
sys.path.append("../")
from utils.experiment_utils import mean_dfs
from utils.utils_base import *
from itertools import *
from utils.utils_base import *
from utils.data_loader import *
from method.diffnaps import *
from utils.data_loader import  perm
from utils.gen_synth_datasets import *
from utils.experiment_utils import *
from utils.measures import *
import time
import argparse



auc_list = []
for i in range(10):
    print(i)

    conf_dict = {}

    conf_dict["cardio"] = TrainConfig(hidden_dim=500, epochs=25, weight_decay=8.0, elb_k=1, elb_lamb=1, class_elb_k=5,
                                      class_elb_lamb=5,
                                      lambda_c=100.0, regu_rate=1.08, class_regu_rate=1.08, batch_size=64,
                                      log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,
                                      seed=14401360119984179300,
                                      input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1,
                                      label_dim=2, beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='add2'
                                      )
    conf_dict["cardio"].t1 = 0.15
    conf_dict["cardio"].t2 = 0.1

    conf_dict["disease"] = TrainConfig(hidden_dim=1500, epochs=80, weight_decay=9.0, elb_k=0, elb_lamb=0, class_elb_k=0,
                                       class_elb_lamb=0,
                                       lambda_c=50, regu_rate=1.08, class_regu_rate=1.08, batch_size=32,
                                       log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,
                                       seed=14401360119984179300,
                                       input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1,
                                       label_dim=41,
                                       beta=0.01, hyper_hidden_layers=[512, 1024], device="cuda", mode='add2')
    conf_dict["disease"].t1 = 0.15
    conf_dict["disease"].t2 = 0.1

    conf_dict["brca-n"] = TrainConfig(hidden_dim=10000, epochs=20, weight_decay=5.0, vertical_decay=0.0, elb_k=1,
                                      elb_lamb=1, class_elb_k=1, class_elb_lamb=1,
                                      lambda_c=10.0, regu_rate=1.08, class_regu_rate=1.08, batch_size=32,
                                      log_interval=100, sparse_regu=0, init_enc="", test=False, lr=0.005, seed=0,
                                      model=DiffnapsNet,
                                      input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1,
                                      label_dim=2,
                                      beta=0.01, hyper_hidden_layers=[1024, 1024], device="cuda", mode='add2')
    conf_dict["brca-n"].t1 = 0.02
    conf_dict["brca-n"].t2 = 0.02

    conf_dict["brca-s"] = TrainConfig(hidden_dim=30000, epochs=30, weight_decay=12.0, vertical_decay=0.0, elb_k=1,
                                      elb_lamb=1, class_elb_k=1, class_elb_lamb=1,
                                      lambda_c=200.0, regu_rate=1.08, class_regu_rate=1.08, batch_size=32,
                                      log_interval=100, sparse_regu=0, init_enc="", test=False, lr=0.005, seed=0,
                                      model=DiffnapsNet,
                                      input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1,
                                      label_dim=4,
                                      beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='add2')
    conf_dict["brca-s"].t1 = 0.02
    conf_dict["brca-s"].t2 = 0.03

    conf_dict["genomes"] = TrainConfig(hidden_dim=2000, epochs=100, weight_decay=5, elb_k=10, elb_lamb=10,
                                       class_elb_k=20, class_elb_lamb=20,
                                       lambda_c=25.0, regu_rate=1.1, class_regu_rate=1.1, batch_size=128,
                                       log_interval=100, sparse_regu=0, test=False, lr=0.001, model=DiffnapsNet,
                                       seed=14401360119984179300, init_enc="bimodal",
                                       input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1,
                                       label_dim=6,
                                       beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='add2')
    conf_dict["genomes"].t1 = 0.03
    conf_dict["genomes"].t2 = 0.8

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=str)

    args = parser.parse_args()
    print(args.dataset)

    if args.dataset == "cardio":
        data, labels, translator = load_cardio_vasc()
        label_dict = {0: "healthy", 1: "heartattack"}
    elif args.dataset == "disease":
        data, labels, label_dict, translator = load_disease_symptom_prediction()
        translator = list(translator.values())
    elif args.dataset == "brca-n":
        data, labels, translator = load_brca_bin()
        label_dict = {0: "Adj normal", 1: "tumor"}
    elif args.dataset == "brca-s":
        data, labels, translator = load_brca_mult()
        label_dict = {0: "0", 1: "1", 2: "2", 3: "3"}
        translator = list(translator)
    elif args.dataset == "genomes":
        data, labels, label_dict, translator = load_1000_genomes()

    conf = conf_dict[args.dataset]
    # start_time = time.time()
    # Train diffnaps
    model, new_weights, trainDS, c_w, time_taken = learn_diffnaps_net(data,conf,labels = labels)
    # time_taken = time.time() - start_time
    # time_taken = time_taken / 60
    enc_w = model.fc0_enc.weight.data.detach().cpu()
    # c_w = model.classifier.weight.detach().cpu()
    # extract the differntial patterns, t1 is t_e and t2 is t_c
    _,_,_,num_pat,res_dict, _ = get_positional_patterns(enc_w.detach().cpu(),c_w.detach().cpu(), general=True, t_mean=1.0,  t1=conf.t1,t2=conf.t2)
    res_dict = {int(k):v for k,v in res_dict.items()}
    res_to_csv("diffnaps",args.dataset, res_dict, data, labels, label_dict, translator)
    write_time("diffnaps", args.dataset,time_taken,res_dict)

    _, _, auc = roc(res_dict, data, labels, label_dict, translator, verbose=False)
    print(auc)
    auc_list.append(auc)
    del model
    torch.cuda.empty_cache()
print(auc_list)
