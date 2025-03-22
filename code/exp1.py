import sys
import os
sys.path.append("../")
from utils.experiment_utils import mean_dfs
import numpy as np
import pandas as pd
from utils.utils_base import *

from itertools import *
from utils.utils_base import *
from utils.data_loader import *
from method.diffnaps import *
from utils.data_loader import  perm
from utils.gen_synth_datasets import *
from utils.measures import *
import time

#torch.autograd.set_detect_anomaly(True)

if not os.path.exists("../results/synth_results/exp1/"):
      os.makedirs("../results/synth_results/exp1/")

def gen_configs():
      conf_dict_exp1 = {}

      conf_dict_exp1[100] = TrainConfig(hidden_dim = 250, epochs=25, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=64, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=2, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add2')

      conf_dict_exp1[500] = TrainConfig(hidden_dim = 250, epochs=25, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=64, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                        input_dim=500, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='None')

      conf_dict_exp1[1000] = TrainConfig(hidden_dim = 500, epochs=25, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=64, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                         input_dim=1000, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                         beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='replace')

      conf_dict_exp1[5000] = TrainConfig(hidden_dim = 2000, epochs=25, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=64, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                         input_dim=5000, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001,ngen=1, label_dim=2,
                                         beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='replace')


      conf_dict_exp1[10000] = TrainConfig(hidden_dim = 2000, epochs=25, weight_decay=5, elb_k=0, elb_lamb=2, class_elb_k=2, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet, init_enc="",
                                          input_dim=10000, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                          beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='replace')

      conf_dict_exp1[15000] = TrainConfig(hidden_dim = 2000, epochs=25, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                          input_dim=150000, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                          beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='replace')

      conf_dict_exp1[20000] = TrainConfig(hidden_dim = 2000, epochs=25, weight_decay=1, elb_k=1, elb_lamb=1, class_elb_k=0, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                          input_dim=20000, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                         beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='replace')


      conf_dict_exp1[25000] = TrainConfig(hidden_dim = 3500, epochs=25, weight_decay=5, elb_k=2, elb_lamb=2, class_elb_k=0, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet, init_enc="",
                                          input_dim=25000, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                         beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='replace')


      conf_dict_exp1[50000] = TrainConfig(hidden_dim = 3500, epochs=25, weight_decay=5, elb_k=1, elb_lamb=1, class_elb_k=0, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet, init_enc="",
                                          input_dim=50000, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                        beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='replace')


      conf_dict_exp1[75000] = TrainConfig(hidden_dim = 3500, epochs=25, weight_decay=5, elb_k=1, elb_lamb=1, class_elb_k=10, class_elb_lamb=10,
                                          lambda_c = 15, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet, init_enc="",
                                          input_dim=75000, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                        beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='replace')


      conf_dict_exp1[100000] = TrainConfig(hidden_dim = 4000, epochs=25, weight_decay=8, elb_k=5, elb_lamb=5, class_elb_k=10, class_elb_lamb=10,
                                    lambda_c = 30, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                                    log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet, init_enc="",
                                        input_dim=100000, z=64, s=256, bias=False, k=5, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                        beta=0.01, hyper_hidden_layers=[512, 512], device="cuda", mode='replace')


      return conf_dict_exp1

dfs = []
for seed in [0,1,2]:
      exp_dict = experiment1([ 100],seed=seed) #[ 100, 500, 1000, 5000, 10000, 15000,20000,25000,50000,1000000]
      avg_res_list = []
      conf_dict_exp1 = gen_configs()
      for cols in exp_dict.keys():
            print(seed,cols)
            data, labels, gt_dict = exp_dict[cols]
            data, labels = perm(data.astype(np.float32),labels.astype(int))
            gt_dict = {str(k):v for k,v in gt_dict.items()}

            # start_time = time.time()
            # Train diffnaps
            model, new_weights, trainDS, c_w, time_taken, test_loader  = learn_diffnaps_net(data, conf_dict_exp1[cols], labels = labels,ret_test=True,verbose=True)
            # time_taken = time.time() - start_time
            # time_taken = time_taken / 60

            # enc_w,c_w = model.get_best_weights()
            enc_w = model.fc0_enc.weight.data.detach().cpu()
            # c_w = model.get_classifier_weight().detach().cpu()
            print("Encoder mean: ", enc_w.mean())
            print("Encoder: ", enc_w.std())


            enc_bin = 0.3
            class_bin = 0.3
            # extract the differntial patterns, t1 is t_e and t2 is t_c 
            _,_,_,num_pat,patterns, gen_patterns = get_positional_patterns(enc_w.detach().cpu(), c_w.detach().cpu(), general=True, t_mean=1,  t1=enc_bin, t2=class_bin)

            jd = mean_overlap_function(patterns, gt_dict)
            sp, sr, f1 = mean_compute_scores(patterns, gt_dict)

            avg_res_list.append([cols, jd, sp, sr, f1, time_taken] )
            print(cols, f1)
            del model


      res = np.array(avg_res_list)
      df = pd.DataFrame(res,columns=["ncols","JD","SP","SR","F1","time"])
      df.to_csv("../results/synth_results/exp1/exp1_diffnaps_%d.csv"%seed,index=False)
      dfs.append(df)
mean_dfs(dfs, "exp1", "exp1_diffnaps")
