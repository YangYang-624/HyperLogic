from itertools import *
import tabulate
import numpy as np

def overlab_measure(pat,gt):
    nom = len(set(pat).intersection(set(gt)))
    denom = len(set(pat).union(set(gt)))
    measure = nom/denom
    return measure

def inner_metric(pat,gt):
    nom = len(set(pat).intersection(set(gt)))
    denom = len(set(gt))
    measure = nom/denom
    return measure

def prec(pat,gt):
    nom = len(set(pat).intersection(set(gt)))
    denom = len(set(pat))
    measure = nom/denom
    return measure

def soft_prec(P_d, P_g):
    total = []
    for p_d in P_d:
        argmax = np.max(list(map(lambda x: overlab_measure(*x), product([p_d],P_g))))
        total.append(argmax)
    return np.mean(total)

def soft_rec(P_d,P_g):
    total = []
    if len(P_d) == 0:
        return 0.000001
    for p_g in P_g:
        argmax = np.max(list(map(lambda x: overlab_measure(*x), product(P_d,[p_g]))))
        total.append(argmax)
    return np.mean(total)


def mean_compute_scores(mining,gt):
    sp_list = []
    sr_list = []
    f1_list = []
    for label in mining.keys():
        P_d = mining[label]
        P_g = gt[label]
        if len(P_d) == 0:
            sp,sr,f1 = 0,0,0
        else:
            sp = soft_prec(P_d, P_g)
            sr = soft_rec(P_d, P_g)
            if sp+sr==0:
                f1 = 0
            else:
                f1 = (2*sp*sr)/(sp+sr)
        sp_list.append(sp)
        sr_list.append(sr)
        f1_list.append(f1)
    return np.mean(sp_list), np.mean(sr_list), np.mean(f1_list)

def compute_scores(mining,gt):
    score_dict = {}
    for label in mining.keys():
        P_d = mining[label]
        P_g = gt[label]
        if len(P_d) == 0:
            score_dict[label] = {"SP":0, "SR":0, "F1":0}
        else:
            sp = soft_prec(P_d, P_g)
            sr = soft_rec(P_d, P_g)
            if sp+sr==0:
                f1 = 0
            else:
                f1 = (2*sp*sr)/(sp+sr)
            score_dict[label] = {"SP":sp, "SR":sr, "F1":f1}
    return score_dict



def overlap_function(mining, gt):
    scores = {}
    for k2, v2 in mining.items():
        res = {}
        res = {i:{"index": -1, "overlap":-1.0} for i in range(len(gt[k2]))}
        res["Avg Overlap"] = 0.0
        if len(v2)>0:
            acc_overlaps = 0.0
            for i, pat in enumerate(gt[k2]):
                
                overlaps = list(map(lambda x: overlab_measure(*x), product([pat],v2)))
                pos = np.argmax(overlaps)
                max_overlap = np.max(overlaps)
                if max_overlap > 0.0:
                    res[i] = {"index": pos, "overlap":max_overlap}
                    acc_overlaps += max_overlap
                else:
                    res[i] = {"index": -1, "overlap":-1.0}
            res["Avg Overlap"] = acc_overlaps/len(gt[k2])
        scores[k2] = res
    return scores

def mean_overlap_function(mining, gt):
    scores = overlap_function(mining, gt)
    eval_list = []
    for k,v in scores.items():
        eval_list.append(v["Avg Overlap"])
    return np.mean(eval_list)




