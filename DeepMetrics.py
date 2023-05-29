"""PAD Metrics"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc


# metrics
def pred_metrics(gt, pred):
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    return tp, tn, fp, fn, tpr, fpr


# PAD metrics
def pad_metrics(gt, preds, ths):
    '''
    Attack Presentation Classification Error Rate (APCER) – 
    the proportion of presentation attack samples incorrectly classified as bona fide presentation
    
    Bona Fide Presentation Classification Error Rate (BPCER) –
    the proportion of bona fide samples incorrectly classified as presentation attack samples
    
    ISO-30107-3:2023
    
    N[PAIS]: is the number of attack presentations for the given PAI species;
    
    Res[i]: takes value 1 if the [i]th presentation is classified as an attack presentation,
        and value 0 if classified as bona fide presentation;
        
    N[BF]: is the numbre of bona fide presentations;
    '''
    
    indx_PAIS = gt == 1
    indx_BF = gt == 0
    NPAIS = np.sum(indx_PAIS)
    NBF = np.sum(indx_BF)
    print('N[PAIS]:', NPAIS)
    print('N[BF]:', NBF)
    
    bpcer_list = []
    apcer_list = []
    auc_list = []
    tpr_list = []
    fpr_list = []
    for th in ths:
        # threshold, operation point
        output = np.zeros(preds.shape[0]) # 0 if bonafide
        output[preds > th] = 1 # 1 if attack presentation
        
        # APCER
        Res_PAIS = np.sum(output[indx_PAIS])
        APCER = 1 - ( ( 1 / NPAIS ) * Res_PAIS )
        
        # BPCER
        Res_BF = np.sum(output[indx_BF])
        BPCER = Res_BF / NBF
        
        tp, tn, fp, fn, tpr, fpr = pred_metrics(gt, output)
        fpr_, tpr_, thresholds = roc_curve(gt, output, pos_label=1)
        auc_ = auc(fpr_, tpr_)
        f1 = round(f1_score(gt, output, average='binary'),3)
        
        print(th, 'auc', round(auc_,3), 'f1',
              round(f1,3), 'bpcer', round(BPCER,3),
              'apcer', round(APCER,3))
        
        bpcer_list.append(BPCER)
        apcer_list.append(APCER)
        auc_list.append(auc_)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return bpcer_list, apcer_list, auc_list, tpr_list, fpr_list
    
