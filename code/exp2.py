import sys
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

if not os.path.exists("../results/synth_results/exp2/"):
      os.makedirs("../results/synth_results/exp2/")

def gen_configs():
      conf_dict_exp2 = {}
      conf_dict_exp2[2] = TrainConfig(hidden_dim = 500, epochs=50, weight_decay=2, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                               lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=32, wd_class=0.0,
                         log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=2,
                                        beta=0.01, hyper_hidden_layers=[512,512], device="cuda", mode='add1')

      conf_dict_exp2[5] = TrainConfig(hidden_dim = 1000, epochs=50, weight_decay=2, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                          lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                                    log_interval=100000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=5,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')

      conf_dict_exp2[10] = TrainConfig(hidden_dim = 2000, epochs=50, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                    lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                              log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=10,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')

      conf_dict_exp2[15] = TrainConfig(hidden_dim = 2000, epochs=50, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                    lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                              log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=15,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')

      conf_dict_exp2[20] = TrainConfig(hidden_dim = 2500, epochs=50, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                               lambda_c = 20, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                         log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=20,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')
      
      conf_dict_exp2[25] = TrainConfig(hidden_dim = 2500, epochs=50, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                    lambda_c = 20, regu_rate=1.08, class_regu_rate=1.08, batch_size=128, wd_class=0.0,
                              log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=25,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')

      conf_dict_exp2[30] = TrainConfig(hidden_dim = 2500, epochs=50, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=1.5, class_elb_lamb=1.5,
                                    lambda_c = 20, regu_rate=1.08, class_regu_rate=1.08, batch_size=256, wd_class=0.0,
                              log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=3, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=30,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')

      conf_dict_exp2[35] = TrainConfig(hidden_dim = 2500, epochs=50, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=1.5, class_elb_lamb=1.5,
                                    lambda_c = 20, regu_rate=1.08, class_regu_rate=1.08, batch_size=256, wd_class=0.0,
                              log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=35,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')

      conf_dict_exp2[40] = TrainConfig(hidden_dim = 3000, epochs=50, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=1.5, class_elb_lamb=1.5,
                                    lambda_c = 30, regu_rate=1.08, class_regu_rate=1.08, batch_size=256, wd_class=0.0,
                              log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=40,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')

      conf_dict_exp2[45] = TrainConfig(hidden_dim = 4000, epochs=50, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=1.5, class_elb_lamb=1.5,
                                    lambda_c = 50, regu_rate=1.08, class_regu_rate=1.08, batch_size=256, wd_class=0.0,
                              log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=45,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')

      conf_dict_exp2[50] = TrainConfig(hidden_dim = 4000, epochs=50, weight_decay=1, elb_k=0, elb_lamb=0, class_elb_k=1, class_elb_lamb=1,
                                    lambda_c = 50, regu_rate=1.08, class_regu_rate=1.08, batch_size=256, wd_class=0,
                              log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,init_enc="",
                                        input_dim=100, z=64, s=256, bias=False, k=30, hyper_lr=0.0001, wd=0.0001, ngen=1, label_dim=50,
                                        beta=0.01, hyper_hidden_layers=[512,512],device="cuda", mode='add1')

      return conf_dict_exp2

dfs = []
classes = [2,5,10,15,20,25,30,35,40,45,50]
for seed in [0,1,2,3,4]:
      conf_dict_exp2 = gen_configs()
      exp_dict = experiment2(classes,seed)
      avg_res_list = []

      for k in exp_dict.keys():
            print(10*"#",k,seed,10*"#")
            data, labels, gt_dict = exp_dict[k]
            data, labels = perm(data.astype(np.float32),labels.astype(int))
            gt_dict = {str(k):v for k,v in gt_dict.items()}
            # start_time = time.time()
            # Train diffnaps
            model, new_weights, trainDS, c_w, time_taken, test_loader = learn_diffnaps_net(data, conf_dict_exp2[k], labels = labels,ret_test=True)
            # time_taken = time.time() - start_time
            # time_taken = time_taken / 60

            enc_w = model.fc0_enc.weight.data.detach().cpu()
            print("Encoder mean: ",enc_w.mean())
            print("Encoder: ",enc_w.std())
            # c_w = model.classifier.weight.detach().cpu()


            # extract the differntial patterns, t1 is t_e and t2 is t_c 
            _,_,_,num_pat,patterns, gen_patterns = get_positional_patterns(enc_w.detach().cpu(),c_w.detach().cpu(), general=True, t_mean=1,  t1=0.3,t2=0.3)

            jd = mean_overlap_function(patterns, gt_dict)
            sp, sr, f1 = mean_compute_scores(patterns,gt_dict)

            avg_res_list.append([k, jd, sp, sr, f1, time_taken] )
            print(k, f1)
            del model
      res = np.array(avg_res_list)
      df = pd.DataFrame(res,columns=["ncols","JD","SP","SR","F1","time"])
      df.to_csv("../results/synth_results/exp2/exp2_diffnaps_%d.csv"%seed,index=False)
      dfs.append(df)
mean_dfs(dfs, "exp2", "exp2_diffnaps")