import json
import os
import shutil
import re
from tqdm import tqdm
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score,confusion_matrix
import numpy as np

def calculate_precision(labels, preds):
    """计算precision"""
    true_positives = 0
    false_positives = 0
    if len(labels) == 0 and len(preds) == 0:
        return None
    for pred in preds:
        if pred in labels:
            true_positives += 1
        else:
            false_positives += 1
    return true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0

def calculate_recall(labels, preds):
    """计算recall"""
    total = len(labels)
    pred_true = []
    if len(labels) == 0 and len(preds) == 0:
        return None
    for pred in preds:
        if pred in labels and pred not in pred_true:
            pred_true.append(pred)
    return len(pred_true) / total if total > 0 else 0

def calculate_f1_score(labels, preds):
    """计算f1_score"""
    precision = calculate_precision(labels, preds)
    recall = calculate_recall(labels, preds)
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

def calculate_performance(info_paths:list,label_names:list):

    label_list = []
    predict_list = []
    

    for idx,info_path in enumerate(info_paths):
        with open(info_path,'r',encoding='utf-8') as f:
            predict_info = json.load(f)


        for info in tqdm(predict_info['all_results']):
            predict = info['final_decision']
            predict_int = classes_name.index(predict)
            label_int = classes_name.index(label_names[idx])
        
            predict_list.append(predict_int)
            label_list.append(label_int)

    # 准确率 
    print('-----------------------------'*3)
    acc = accuracy_score(label_list, predict_list)

    # 精确率
    precision = precision_score(label_list, predict_list, zero_division=0,average=None)
    # 召回率
    recall = recall_score(label_list, predict_list, zero_division=0,average=None)
    # f1-score
    f1 = f1_score(label_list, predict_list, zero_division=0,average=None)
    
    # 计算混淆矩阵
    cm = confusion_matrix(label_list, predict_list)
    print(f'混淆矩阵:\n{cm}')
    print('-----------------------------'*3)
    
    print(f'acc: {acc:.4f}')
    for class_id,class_name in enumerate(classes_name):
        print(f'{class_name}\n\tprecision: {precision[class_id]:.4f}\n\trecall: {recall[class_id]:.4f}\n\tf1_score: {f1[class_id]:.4f}')

    # 计算每个类别的TP、TN、FP、FN
    print('\n' + '='*50)
    print('每个类别的混淆矩阵指标:')
    print('='*50)
    
    for class_id, class_name in enumerate(classes_name):
        # 对于多分类问题，将当前类作为正类，其他类作为负类
        tp = cm[class_id, class_id]  # 真正例：实际为该类且预测为该类
        fn = np.sum(cm[class_id, :]) - tp  # 假负例：实际为该类但预测为其他类
        fp = np.sum(cm[:, class_id]) - tp  # 假正例：实际为其他类但预测为该类
        tn = np.sum(cm) - tp - fn - fp  # 真负例：实际为其他类且预测为其他类
        
        print(f'\n【{class_name}】:')
        print(f'  TP (真正例): {tp}')
        print(f'  TN (真负例): {tn}')
        print(f'  FP (假正例): {fp}')
        print(f'  FN (假负例): {fn}')
        print(f'  总样本数: {tp + tn + fp + fn}')




if __name__ == "__main__":
    classes_name = ["盖板缺失","盖板存在"]
    predict_paths = ["inference_result/complete_directory_test_20250715_014557.json","inference_result/complete_directory_test_20250715_022547.json"]
    label_names = ["盖板缺失","盖板存在"] # 指定当前验证集的正确标签，需要与predict_paths的一一对应
    assert len(predict_paths) == len(label_names)
    calculate_performance(predict_paths,label_names)  
        
        