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

def gen_configs():
      conf = TrainConfig(hidden_dim = 500, epochs=50, weight_decay=2, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                               lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=32, wd_class=0.0,
                         log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="")
      return conf

dfs = []
snrs = [0,10,20,30,40,50,60,70,80,90,100]
for seed in [0,1,2,3,4]:
      conf_exp3 = gen_configs()
      exp_dict = experiment3(snrs,seed=seed)
      avg_res_list = []
      for k in exp_dict.keys():
            print(10*"#",k,seed,10*"#")
            data, labels, gt_dict = exp_dict[k]
            data, labels = perm(data.astype(np.float32),labels.astype(int))
            gt_dict = {str(k):v for k,v in gt_dict.items()}
            start_time = time.time()
            # Train diffnaps
            model, new_weights, trainDS, test_loader = learn_diffnaps_net(data, conf_exp3, labels = labels,ret_test=True)
            time_taken = time.time() - start_time
            time_taken = time_taken / 60

            enc_w = model.fc0_enc.weight.data.detach().cpu()
            print("Encoder mean: ",enc_w.mean())
            print("Encoder: ",enc_w.std())
            c_w = model.classifier.weight.detach().cpu()

            enc_bin = 0.3
            class_bin = 0.3
            # extract the differntial patterns, t1 is t_e and t2 is t_c 
            _,_,_,num_pat,patterns, gen_patterns = get_positional_patterns(enc_w,c_w, general=True, t_mean=1,  t1=enc_bin,t2=class_bin)

            jd = mean_overlap_function(patterns, gt_dict)
            sp, sr, f1 = mean_compute_scores(patterns,gt_dict)
            print(k, f1)
            avg_res_list.append([k, jd, sp, sr, f1, time_taken] )
      res = np.array(avg_res_list)
      df = pd.DataFrame(res,columns=["ncols","JD","SP","SR","F1","time"])
      df.to_csv("../results/synth_results/diffnaps/exp3_diffnaps_%d.csv"%seed,index=False)
      dfs.append(df)
mean_dfs(dfs, "diffnaps", "exp3_diffnaps")