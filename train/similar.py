from smart import fp_search as fp
import pubchempy as pcp
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import json
import torch

input_file = './file/input_file_challenge2016_predict_p3.txt'

def replacex(smile):
    smile = smile.replace('Cl', 'X')
    smile = smile.replace('Br', 'Y')
    smile = smile.replace('Si', 'G')
    smile = smile.replace('Se', 'Z')
    smile = smile.replace('10', '0')
    smile = smile.replace('Na', 'M')
    smile = smile.replace('Fe','T')
    smile = smile.replace('K','T')
    smile = smile.replace('Au','T')
    smile = smile.replace('As','T')
    smile = smile.replace('Cr','T')
    smile = smile.replace('Al','T')
    return smile

def replace_smile(smile):
    smile = replacex(smile)
    smile = smile.replace('(','\(')
    smile = smile.replace(')','\)')
    smile = smile.replace('[','\[')
    smile = smile.replace(']','\]')
    smile = smile.replace('+','\+')
    smile = smile.replace('.','\.')
    return smile

def look_up(formula):
    total = []
    total_temp = []
    formula = formula.replace('G','Si')
    input_file = './candidate/PubChem_compound_formulaquery_{0}.json'.format(formula)
    with open(input_file, 'r', encoding='utf-8') as fw:
        injson = json.load(fw)
        for each in injson['IsomericSMILES']:
            total_temp.append(each)
        for i in total_temp:
            total.append(injson['IsomericSMILES'][i])
    return total

def final_score(smile,smile_total):
    smile_score = fp()
    fingerprint = smile_score.fun_smile(smile,smile_total)
    score_temp = smile_score.get_score()
    score = torch.mul(fingerprint,score_temp)
    return score,fingerprint

def judge_smile(formula):
    formula = formula.replace('G','Si')
    unfound = []
    if os.path.exists('./candidate/PubChem_compound_formulaquery_{0}.json'.format(formula)):
        pass
    else:
        try:  
            df3 = pcp.get_properties(['isomeric_smiles'], formula, 'formula', as_dataframe=True)
            df3.to_json('./candidate/PubChem_compound_formulaquery_{0}.json'.format(formula))
        except:
            unfound.append(formula + '\n')

def fingerprint_dict(formula):
    if os.path.exists('./candidate/{0}.txt'.format(formula)):
        fingerprint_dict = {}
        with open('./candidate/{0}.txt'.format(formula),'r') as f:
            for line in f.readlines():
                smile_temp1 = line.strip().split(',')[1]
                fp_temp1 = line.strip().split(',')[2:]
                fingerprint_dict[smile_temp1] = fp_temp1
    else:
        candidate = look_up(formula)
        candidate = [replacex(i) for i in candidate]
        smile_temp = fp()
        dict_t,smile_total = smile_temp.search_smile()
        fp_list = []
        for i,element in enumerate(candidate):           
            _,fp_temp2 = final_score(element,smile_total)
            fp_list.append(str(i)+','+str(element)+',')
            for i,element in enumerate(fp_temp2):
                if i + 1 == len(fp_temp2):
                    fp_list.append(str(int(element)))
                else:
                    fp_list.append(str(int(element)) + ',')
            fp_list.append('\n')
        with open('./candidate/{0}.txt'.format(formula),'w') as f:
            for i in fp_list:
                f.write(i)
        if os.path.exists('./candidate/{0}.txt'.format(formula)):
            fingerprint_dict = {}
            with open('./candidate/{0}.txt'.format(formula),'r') as f:
                for line in f.readlines():
                    smile_temp1 = line.strip().split(',')[1]
                    fp_temp1 = line.strip().split(',')[2:]
                    fingerprint_dict[smile_temp1] = fp_temp1
        else:
            print('there is an error need to be fixed out')
    return fingerprint_dict

def calculation(formula,pre,trg):
    t0 = 0
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    t6 = 0
    t7 = 0
    t8 = 0
    t9 = 0
    t10 = 0
    judge_smile(formula)
    fingerprint_dict1 = fingerprint_dict(formula)
    
    candidate = [i for i in fingerprint_dict1]
    if pre in candidate and pre == trg:             
        t0 = 1
    else:
        score_weight = fp()
        _,smile_total = score_weight.search_smile()
        score_w = score_weight.get_score()   
        score_pre,_ = final_score(pre,smile_total)
        score_list = []
        top1 = []
        top2 = []
        top3 = []
        top4 = []
        top5 = []
        top6 = []
        top7 = []
        top8 = []
        top9 = []
        top10 = []
        for i in candidate:
            fingerprint_i = torch.tensor(list(map(float,fingerprint_dict1[i])))
            score_i = torch.mul(fingerprint_i,score_w)
            score = torch.dot(score_pre, score_i)
            score_list.append(score)
        top1_number = np.flipud(np.argsort(score_list)[-1:])
        top2_number = np.flipud(np.argsort(score_list)[-3:])
        top3_number = np.flipud(np.argsort(score_list)[-4:])
        top4_number = np.flipud(np.argsort(score_list)[-5:])
        top5_number = np.flipud(np.argsort(score_list)[-5:])
        top6_number = np.flipud(np.argsort(score_list)[-6:])
        top7_number = np.flipud(np.argsort(score_list)[-7:])
        top8_number = np.flipud(np.argsort(score_list)[-8:])
        top9_number = np.flipud(np.argsort(score_list)[-9:])
        top10_number = np.flipud(np.argsort(score_list)[-10:])
        for i in top1_number:
            top1.append(candidate[i])
        for i in top2_number:
            top2.append(candidate[i])
        for i in top3_number:
            top3.append(candidate[i])
        for i in top4_number:
            top4.append(candidate[i])
        for i in top5_number:
            top5.append(candidate[i])
        for i in top6_number:
            top6.append(candidate[i])
        for i in top7_number:
            top7.append(candidate[i])
        for i in top8_number:
            top8.append(candidate[i])
        for i in top9_number:
            top9.append(candidate[i])
        for i in top10_number:
            top10.append(candidate[i])
        if trg in top1:
            t1 = 1
        if trg in top2:
            t2 = 1
        if trg in top3:
            t3 = 1
        if trg in top4:
            t4 = 1
        if trg in top5:
            t5 = 1
        if trg in top6:
            t6 = 1
        if trg in top7:
            t7 = 1
        if trg in top8:
            t8 = 1
        if trg in top9:
            t9 = 1
        if trg in top10:
            t10 = 1
    return t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10

def accuracy():
    with open(input_file,'r') as f:
        at0 = 0
        at1 = 0
        at2 = 0
        at3 = 0
        at4 = 0
        at5 = 0
        at6 = 0
        at7 = 0
        at8 = 0
        at9 = 0
        at10 = 0
        n1 = 0
        result_list = []
        for line in tqdm(f.readlines()):
            line = line.strip().split(':')
            if 'compound' in line[0]:
                n1 += 1
                num = int(line[0].split('_')[1])
                formula = line[1]
                x = 2
            if x == 1:
                trg = line[1]
            if x == 0:
                    pre = line[1]
                    at0_temp,at1_temp,at2_temp,at3_temp,at4_temp,at5_temp,at6_temp,at7_temp,at8_temp,at9_temp,at10_temp = calculation(formula,pre,trg)
                    at0 += at0_temp
                    at1 += at1_temp
                    at2 += at2_temp
                    at3 += at3_temp
                    at4 += at4_temp
                    at5 += at5_temp
                    at6 += at6_temp
                    at7 += at7_temp
                    at8 += at8_temp
                    at9 += at9_temp
                    at10 += at10_temp
                    "判断每一个是否正确"
                    resultlisttemp = [at0_temp,at1_temp,at2_temp,at3_temp,at4_temp,at5_temp,at6_temp,at7_temp,at8_temp,at9_temp,at10_temp]
                    if sum(resultlisttemp) == 0:
                        result_list.append(str(num)+':'+formula+':'+trg+':'+pre+':'+'100')
                    else:
                        for i,element in enumerate(resultlisttemp):
                            if element == 1:
                                result_list.append(str(num)+':'+formula+':'+trg+':'+pre+':'+str(i))
                                break
                        
            x -= 1
    accuracy_rate = [at0,at0+at1,at0+at2,at0+at3,at0+at4,at0+at5,at0+at6,at0+at7,at0+at8,at0+at9,at0+at10]
    for i in accuracy_rate:
        print(i,round((i/n1*100),2))
    return result_list

result_list = accuracy()
