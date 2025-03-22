import sys
sys.path.append("../")
import os
from utils.data_loader import split_dataset
import pandas as pd
from utils.measures import *
import numpy as np
import json


#################### General #####################

def print_to_csv(res_dict, exp_dict, method, exp,ascending=True,write=True):
    avg_res_list = []
    for k, (patterns,time_taken) in res_dict.items():
        gt_dict = exp_dict[k][2]
        jd = mean_overlap_function(patterns, gt_dict)
        sp, sr, f1 = mean_compute_scores(patterns,gt_dict)
        avg_res_list.append([k, jd, sp, sr, f1, time_taken] )

    res = np.array(avg_res_list)
    df = pd.DataFrame(res,columns=["ncols","JD","SP","SR","F1","time"])
    df = df.sort_values("ncols",ascending=ascending).reset_index(drop=True)
    if write:
        df.to_csv("./experiments/synth_results/%s/%s.csv"%(method,exp),index=False)
    return df



def mean_dfs(dfs, method, exp):
    final_df = []
    for col in dfs[0].columns[1:]:
        cols = []
        for i,df in enumerate(dfs):
            cols.append(df[col].rename(col+"_%d"%i, inplace=True))
        df_new = pd.DataFrame(cols).transpose() 

        final_df.append(df_new.mean(axis=1).rename(col+"_mean", inplace=True))
        final_df.append(df_new.std(axis=1).rename(col+"_std", inplace=True))
        final_df.append(df_new.max(axis=1).rename(col+"_max", inplace=True))
        final_df.append(df_new.min(axis=1).rename(col+"_min", inplace=True))

    final_df = pd.DataFrame(final_df).T.set_index(dfs[0].iloc[:,0])
    print(final_df.iloc[:,1])
    final_df.to_csv("../results/synth_results/%s/%s.csv"%(method,exp),index=True)
    return  final_df

def translate_dict(res_dict, translator, label_dict):
    trans = lambda x:translator[x]
    translated_dict = {}
    for k,patterns in res_dict.items():
        translated_patterns = [ list(map(trans,pattern)) for pattern in patterns]
        translated_dict[label_dict[k]] = translated_patterns
    return translated_dict

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def res_to_csv(method, dataname, res_dict, data,labels,label_dict,translator,verbose=False):
    labels = np.array(labels)
    folder = os.path.join("../results/real_results/",method)
    print(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    splitted_dataset = split_dataset(data, labels, label_dict,keep=True)
    tsv = "Label \t Pattern \t Pattern_num \t P(class|pattern) \t Supp in class \t Supp in data \n"
    for label_data, data_part in splitted_dataset.items():
        print("#"*10 + str((label_dict[label_data])) + "#"*10)
        print("Iter",label_data)
        if not label_data  in res_dict.keys():
            continue
        for pat in res_dict[label_data]:
            supp_1 = np.sum(np.sum(data_part[:,pat],axis=1)==(len(pat)))
            supp_2 = np.sum(np.sum(data[:,pat],axis=1)==(len(pat)))
            rel_supp_1 = supp_1/data_part.shape[0]*100
            rel_supp_2 = supp_2/(data.shape[0])*100
            if rel_supp_2>0.0001:
                odds = rel_supp_1/rel_supp_2
            else:
                odds = 0.0
            if True:
                hit = labels[np.sum(data[:,pat],axis=1)==(len(pat))]==label_data
                if len(hit) > 1 and round(np.sum(hit)/len(hit)*100) > 0:
                    translated = list(map(lambda x: translator[x], pat))
                    if verbose:
                        print(translated)
                        print("Support in class: ", round(rel_supp_1,4),"%")
                        print("Support in data: ",round(rel_supp_2,4),"%")
                        print("Odds: ", round(odds,2))
                        print("P(class=%s|pattern) = "%(label_dict[label_data]), round(np.sum(hit)/len(hit)*100,2), "%")
                        print()
                    strr = label_dict[label_data] + "\t" + str(translated) + "\t" + str(pat)+ "\t" +str(round(np.sum(hit)/len(hit)*100,2))+ "\t" +str(round(rel_supp_1,4)) + "\t" +str(round(rel_supp_2,4))  +"\n" 
                    tsv +=  strr
    print(res_dict.keys())
    with open(os.path.join(folder,dataname+".json"),"w") as f:
        json.dump(res_dict, f, indent = 6,cls=NpEncoder)

    with open(os.path.join(folder,dataname+".tsv"),"w") as f:
        f.write(tsv)
            
def write_time(method, dataname, time_taken,res_dict):
    folder = os.path.join("../results/real_results/",method)
    total_patterns = np.sum([ len(v) for k,v in res_dict.items()] )
    with open(os.path.join(folder,dataname+"_time.txt"),"w") as f:
        f.write(str(time_taken)+","+str(total_patterns))

def translate_back(patterns, translator):
    trans_patterns = []
    
    for p in patterns:
        p = list(map(int,p[1:-1].split(",")))
        trans_patterns.append(p)
    return trans_patterns
            

def roc(res_dict, data,labels,label_dict,translator,verbose=False):
    if isinstance(res_dict,str):
        with open(res_dict) as f:
            res_dict = json.load(f)
    res_dict = {int(k):v for k,v in res_dict.items()}
    labels = np.array(labels)
    splitted_dataset = split_dataset(data, labels, label_dict,keep=True)
    max_label = np.max(labels) + 1 
    specificities = []
    for label_data, data_part in splitted_dataset.items():
        if not label_data  in res_dict.keys():
            continue
        spec = []
        line_x = np.arange(0,1.05,0.05)
        for ratio in line_x:
            filtered = []
            hitts = np.zeros(data_part.shape[0])
            for pat in res_dict[label_data] :
                
                supp_1 = np.sum(np.sum(data_part[:,pat],axis=1)==(len(pat)))
                supp_2 = np.sum(np.sum(data[:,pat],axis=1)==(len(pat)))
                rel_supp_2 = supp_2/(data.shape[0])*100
                if supp_1 < 1:
                    continue
                if rel_supp_2>0.0001:
                    odds = supp_1/supp_2
                else:
                    odds = 0.0
                if odds > ratio:
                    mask = np.sum(data[:,pat],axis=1)==(len(pat))
                    correct = (labels[mask] == label_data)
                    acc = np.sum(correct)/len(correct)
                    if acc > (1/max_label+0.1) and len(pat)>0:
                        hitts = np.logical_or(hitts,np.sum(data_part[:,pat],axis=1)==(len(pat)))
                        filtered.append(hitts)
            spec.append(np.mean(hitts))

        specificities.append(spec)

    specificities = np.array(specificities)
    line_y = specificities.mean(axis=0)
    auc = np.mean(line_y)

    return line_x, line_y, auc


def real_odds(res_dict, data,labels,label_dict,translator,verbose=False):
    if isinstance(res_dict,str):
        with open(res_dict) as f:
            res_dict = json.load(f)
    labels = np.array(labels)

    splitted_dataset = split_dataset(data, labels, label_dict,keep=True)
    tsv = "Label \t Pattern \t Pattern_num \t P(class|pattern) \t Supp in class \t Supp in data \n"
    overall_odds = []
    res_dict = {int(k):v for k,v in res_dict.items()}
    for label_data, data_part in splitted_dataset.items():
        if not label_data  in res_dict.keys():
            continue
        for pat in res_dict[label_data]:
            supp_1 = np.sum(np.sum(data_part[:,pat],axis=1)==(len(pat)))
            supp_2 = np.sum(np.sum(data[:,pat],axis=1)==(len(pat)))
            rel_supp_1 = supp_1/data_part.shape[0]*100

            odds_nom = rel_supp_1
            odds_denom = (supp_2-supp_1)/(data.shape[0]-data_part.shape[0])
            if odds_nom == 0:
                continue
            overall_odds.append(np.log(odds_nom/(odds_denom+0.001)))
    return np.mean(overall_odds)
