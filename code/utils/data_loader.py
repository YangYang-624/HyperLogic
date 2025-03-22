from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
from itertools import *
from torch.optim.lr_scheduler import MultiStepLR
import json
import numpy as np
from tabulate import tabulate
from collections import Counter
from utils.utils_base import *
from utils.gen_synth_datasets import collaps


def selector(data,labels,translator, classes):
    indicator = np.array([ l in classes for l in labels])
    labels = labels[indicator]
    data = data[indicator]
    unique_labels = list(np.unique(labels))
    label_map = dict(zip(unique_labels,range(len(unique_labels))))
    new_labels = np.array([label_map[l] for l in labels])
    new_translator = { new:translator[old] for old,new in label_map.items()}
    return data, new_labels, new_translator

def split_dataset(data, labels, label_dict,keep=False):
    unique_labels = list(np.unique(labels))
    ret = {}
    for l in unique_labels:
        temp = data[labels == l]
        if keep:
            ret[l] = temp
        else:
            ret[label_dict[l]] = temp
    return ret




def load_1000_genomes_dataset():
    super_labels_translator = {'GBR': "EUR",
         'FIN': "EUR",
         'CHS': "EAS",
         'PUR': "AMR",
         'CDX': "EAS",
         'CLM': "AMR",
         'IBS': "EUR",
         'PEL': "AMR",
         'PJL': "SAS",
         'KHV': "EAS",
         'ACB': "AFR",
         'GWD': "AFR",
         'ESN': "AFR",
         'BEB': "SAS",
         'MSL': "AFR",
         'STU': "SAS",
         'ITU': "SAS",
         'CEU': "EUR",
         'YRI': "AFR",
         'CHB': "EAS",
         'JPT': "EAS",
         'LWK': "AFR",
         'ASW': "AFR",
         'MXL': "AMR",
         'TSI': "EUR",
         'GIH': "SAS"}

    colpath = "../data/1kgenomes_variants_af0.01_autosome_genebodyOnly.colnames"
    rowpath = "../data/1kgenomes_variants_af0.01_autosome_promoterOnly.rownames"
    datapath = "../data/1kgenomes_variants_af0.01_autosome_genebodyOnly.dat"
    labelpath = "../data/20130606_sample_info.csv"
    with open(colpath) as f:
        columns = f.readlines()
    with open(rowpath) as f:
        rownames = f.read().split("\n")
    label_df = pd.read_csv(labelpath,sep=";")
    raw_labels = []
    for name in rownames:
        row = label_df[label_df["Sample"]==name]["Population"]
        if len(row)>0:
            raw_labels.append(row.iloc[0])
    label_dict = {}
    for i,k in enumerate(Counter(raw_labels).keys()):
        label_dict[k] = i
    rev_label_dict = {v:k for k,v in label_dict.items()}
    labels = np.array([label_dict[l] for l in raw_labels])
    super_label_dict = {}
    for i,k in enumerate(Counter(list(super_labels_translator.values())).keys()):
        super_label_dict[k] = i
    with open(datapath) as f:
        lines = f.readlines()
        lines = list(map(lambda x: list(map(int,x[:-1].split(" "))), lines))
    max_col = np.max(list(map(max,lines)))
    max_row = len(lines)
    data = np.zeros((max_row, max_col+1),dtype=np.float32)
    for i,line in enumerate(lines):
        data[i,line] = 1
    perm = np.random.permutation(data.shape[0])
    data = data[perm]
    labels = labels[perm]
    columns = columns[0:data.shape[1]]
    columns = np.array(columns)[(data.sum(axis=0)>0)]
    data = data[:,(data.sum(axis=0)>0)]

    super_labels = np.array([super_label_dict[super_labels_translator[l]] for l in raw_labels])
    super_labels = super_labels[perm]
    return data, labels, super_labels, rev_label_dict, super_labels_translator, super_label_dict, columns

def load_1000_genomes():
    np.random.seed(None)
    st = np.random.get_state()[1][0]
    np.random.seed(2147483648)
    data, labels, super_labels, rev_label_dict, super_labels_translator, super_label_dict, columns = load_1000_genomes_dataset()
    rev_super_label_dict = {v:k for k,v in super_label_dict.items()}

    super_pob_codes = {"AFR":"Africans", "AMR": "Admixed Americans", "EAS":"East Asians", "EUR": "Europeans", "SAS": "South Asians"}
    super_pob_codes_int = {v:super_pob_codes[k]  for k,v in super_label_dict.items()} 
    cols = [ col.replace("\n","") for col in columns]
    return data, super_labels,rev_super_label_dict, cols

def compute_bin_labels(bin_edges):
    bin_edges = list(bin_edges)
    bin_edges.insert(0,0)
    labels = list(zip(bin_edges,bin_edges[1:]))[1:]
    #return dict(zip(range(len(labels)), labels))
    return labels

def load_cardio_vasc():
    """
    Age preprocessing
        1. Convert age from days to years
        2. Bin ages into discrete ranges

    Gender preprocessing
        1. Subtract 1

    Height preprocessing
        1. Remove people larger than 200cm or smaller than 140, due to little support
        2. Discretize height in 8 bins using histogram to keep normal dist

    Height preprocessing
        1. Remove outlier i.e w<40 or w>150
        2. Compute bins for cleaned data
        3. Add 2 bins for the outlier

    Blood pressure
        1. https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings

    """
    path = "../data/cardio_train.csv"
    df = pd.read_csv(path, sep=";")
    df["age"] = df["age"]//365
    df["age"],bin_edges = pd.qcut(df["age"], q=6, labels=[0,1,2,3,4,5],retbins=True)
    age_labels = compute_bin_labels(bin_edges)


    df["gender"] = df["gender"] - 1
    gender_labels = ["women", "men"]


    df_less_height = df[df["height"]<200]
    df_less_height = df_less_height[df_less_height["height"]>140]
    df = df_less_height.copy()
    bins = 8
    df_less_height["height"] ,bin_edges = pd.cut(df_less_height["height"], bins=bins, labels=range(bins),retbins=True)
    height_labels = compute_bin_labels(bin_edges)
    #plt.hist(df_less_height["height"], bins=bins)
    #plt.show()

    bins=10
    df_temp = df_less_height[df_less_height["weight"]>40]
    df_temp = df_temp[df_temp["weight"]<150]
    df_1 = df_less_height[df_less_height["weight"]>51.8]
    df_1 = df_1[df_1["weight"]<115]
    a = np.histogram(df_less_height["weight"], bins=bins)
    b = np.histogram(df_temp["weight"], bins=bins)
    c = np.histogram(df_1["weight"], bins=bins-4)
    #print(a[1])
    #print(b[1])
    #print(c[1])
    bins = [0, 52.,62.43333333, 72.86666667, 83.3, 93.73333333, 104.16666667, 120.6,200]
    #plt.hist(df_less["weight"], bins=bins)
    df_less_height_weight = df_less_height.copy()
    df_less_height_weight["weight"],bin_edges = pd.cut(df_less_height["weight"], bins=bins, labels=range(len(bins)-1), retbins=True)
    weight_labels = compute_bin_labels(bin_edges)
    #plt.hist(df_less_height_weight["weight"],bins=8)
    #plt.show()


    #https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings
    df_less_height_weight_api = df_less_height_weight[df_less_height_weight.ap_hi<250]
    df_less_height_weight_api = df_less_height_weight_api[df_less_height_weight_api.ap_hi>50]
    bins=[50,120,129,139,180,250]
    #df_less_height_weight_api["ap_hi_word"] = pd.cut(df_less_height_weight_api["ap_hi"], bins=bins, labels=["Normal", "Elevated", "High_1", "High_2","Crisis"])
    df_less_height_weight_api["ap_hi"] = pd.cut(df_less_height_weight_api["ap_hi"], bins=bins, labels=[0,1,2,3,4])
    #plt.hist(df_less_height_weight_api["ap_hi_word"], bins=5)
    #plt.show()
    #api_labels = dict(zip([0,1,2,3,4], ["Normal", "Elevated", "High_1", "High_2","Crisis"]))
    ap_hi_labels = ["Normal", "Elevated", "High_1", "High_2","Crisis"]


    df_less_height_weight_api_apo = df_less_height_weight_api[df_less_height_weight_api.ap_lo<200]
    df_less_height_weight_api_apo = df_less_height_weight_api_apo[df_less_height_weight_api_apo.ap_lo>40]
    bins =  [39, 80, 89, 120, 200]
    #df_less_height_weight_api_apo["ap_lo_word"] = pd.cut(df_less_height_weight_api_apo["ap_lo"], bins=bins, labels=["Normal_Elevated", "High_1", "High_2","Crisis"])
    df_less_height_weight_api_apo["ap_lo"] = pd.cut(df_less_height_weight_api_apo["ap_lo"], bins=bins, labels=[0,1,2,3])
    #plt.hist(df_less_height_weight_api_apo.ap_lo)
    df_less_height_weight_api_apo.ap_lo.unique()
    #apo_labels = dict(zip([0,1,2,3], ["Normal_Elevated", "High_1", "High_2","Crisis"]))
    ap_lo_labels = ["Normal_Elevated", "High_1", "High_2","Crisis"]



    df_less_height_weight_api_apo.cholesterol = df_less_height_weight_api_apo.cholesterol-1
    cholesterol_labels = ["normal", "above", "way_above"]

    df_less_height_weight_api_apo.gluc = df_less_height_weight_api_apo.gluc-1
    gluc_labels= ["normal", "above", "way_above"]
    smoke_labels=[0,1]
    alco_labels=[0,1]     
    active_labels=[0,1]                                    

    df_processed = df_less_height_weight_api_apo.copy()
    df_processed = df_processed.drop(labels=df_processed.columns[0],axis=1)
    cols = ["age", "gender","height","weight", "ap_hi", "ap_lo", "cholesterol", "gluc","smoke","alco","active"]
    for col in cols:
        content = sorted(df_processed[col].unique())
        labels = locals().get(col+"_labels")
        for k in content:
            col_name = col + "_" + str(labels[k])
            df_processed[col_name] = (df_processed[col]==k).astype(int)
        df_processed = df_processed.drop(col,axis=1)

    labels = df_processed["cardio"].to_numpy()
    df_processed = df_processed.drop("cardio",axis=1)
    data = df_processed.to_numpy()
    data = data.astype(np.float32)
    translator = list(df_processed.columns)
    perm = np.random.permutation(data.shape[0])
    data = data[perm]
    labels = labels[perm]
    return data, labels, translator


def generate_labels(df, col):
    int_labels = df[col]
    labels_values = sorted(list(int_labels.unique()))
    num_to_labels_dict = dict(zip(range(len(labels_values)), labels_values))
    labels_to_num_dict = dict(zip(labels_values, range(len(labels_values))))
    labels = np.array(list(map(lambda x: labels_to_num_dict[x], int_labels)))
    return labels, num_to_labels_dict, labels_to_num_dict

def perm(data,labels):
    perm = np.random.permutation(data.shape[0])
    data = data[perm]
    labels = labels[perm]
    return data, labels

def sparsity(data):
    return data.sum()/data.size
    

def load_disease_symptom_prediction():
    symptom = pd.read_csv("../data/disease_symptom.csv")
    int_labels = symptom["Disease"]
    labels, num_to_labels_dict, labels_to_num_dict = generate_labels(symptom, "Disease")
    data_df = symptom.drop(["Disease"],axis=1)
    values = list(set(collaps(data_df.values)) - set([np.nan]))
    values = sorted(values)
    values_dict = dict(zip(values, range(len(values))))
    data = np.zeros((len(data_df), len(values)))
    for i,row in enumerate(data_df.iterrows()):
        row = row[1]
        row_vals = list(row[~row.isnull()])
        for val in row_vals:
            data[i,values_dict[val]] = 1
    perm = np.random.permutation(data.shape[0])
    data = data[perm].astype(np.float32)
    labels = labels[perm]#.astype(np.float32)
    translator = list(data_df.columns)
    return data, list(labels), num_to_labels_dict, {v:k.strip() for k,v in values_dict.items()}


def load_brca_mult():
    path_data = "../data/brca_processed_logtpm_tumor_noduplicates_75_perc_subtypebalanced.tsv"
    path_flags = "../data/brca_subtype_flag.tsv"    
    data = pd.read_csv(path_data,sep="\t")
    translator = data.columns
    data = data.to_numpy().astype(np.float32)
    labels = (pd.read_csv(path_flags,sep="\t",header=None).to_numpy()).reshape(-1) -1
    return data, labels, translator

def load_brca_bin():
    path_data = "../data/brca_processed_logtpm_balanced_75perc.tsv"
    path_flags = "../data/brca_processed_logtpm_balanced_flag_tissue_origin.tsv"
    data = pd.read_csv(path_data,sep="\t")
    translator = data.columns
    data = data.to_numpy().astype(np.float32)
    labels = (pd.read_csv(path_flags,sep="\t",header=None).to_numpy().reshape(-1) == "Tumor").astype(int)
    return data, labels, translator