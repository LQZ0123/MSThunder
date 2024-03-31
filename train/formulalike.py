from tqdm import tqdm
import pandas as pd
import torch
import re
import numpy as np

input_file = './file/challenge2016-for-p3.txt'
formula_dict = './candidate/dictptof.csv'

def match(input_file):
    total_number = 0
    correction_number = 0
    correction_first_number = 0
    unmatch = []
    with open(input_file,'r') as f:
        score = []
        for line in tqdm(f.readlines()):
            if 'compound' in line:
                precusor = float(line.strip().split(':')[2])
                trg_formula = line.strip().split(':')[1]
                total_number += 1
                n = 2
            if n == 0:
                element_list = line.strip().split(':')[1].split(',')
                element_list = [float(i) for i in element_list]
                element_list = formulalist(element_list)
                cand_formula = look_up(formula_dict,precusor)
                if len(cand_formula) == 0:
                    unmatch.append(trg_formula) 
                    pass
                if len(cand_formula) > 0:
                    if len(cand_formula) == 1:
                        final_formula = cand_formula[0]
                        correction_first_number += 1
                    elif  len(cand_formula) > 1:
                        for i in cand_formula:
                            _,_,cand_tensor = formula_stand(i)
                            o_ = abs(cand_tensor[-1]-element_list[-1])
                            score_o = 0
                            if o_ == 0:
                                score_o = 5
                            elif o_ == 1:
                                score_o = 3
                            elif o_ == 2:
                                score_o = 1
                            elif o_ > 3:
                                score_o = 0 
                            score_element = torch.dot(cand_tensor[:-1],element_list[:-1])
                            '得分为元素之间点乘+氧元素差值为2权重'
                            try:
                                score.append(score_element+score_o)
                            except:
                                import ipdb;ipdb.set_trace()
                        final_formula = cand_formula[int(np.flipud(np.argsort(score)[-1:]))]
                        "这里可能出现最大值相同的情况"
                    if final_formula == trg_formula:
                        correction_number += 1
                    if final_formula != trg_formula:
                        unmatch.append(trg_formula)
                    score = []               
            n -= 1   
    correction_rate = (correction_number/total_number) * 100
    correction_first_number_rate = (correction_first_number/total_number) * 100
    return correction_rate, correction_first_number_rate,unmatch
            

def formulalist(element_list):
    formula = torch.tensor([1,1,0,0,0,0,0,0,0,0,0,0])
    for i,element in enumerate(element_list):
        if i <= 8:
            if element == 27:
                formula[i+2] = 1.0
            elif element == 28:
                formula[i+2] = 0.0
        elif i > 8:
            formula[i+2] = element
    return formula

def look_up(formula_dict,precusor):
    df = pd.read_csv(formula_dict)
    df = dict_slice(df,precusor)
    cand_precusor = []
    cand_formula = []
    unfound = []
    for i,element in enumerate(df['Precusor']):
        if abs((float(precusor)-float(element))*1000000/float(element)) <= 10:
            cand_precusor.append(int(df[df['Precusor']==element]['Number']))
        if len(cand_precusor) == 0:
            unfound.append(precusor)            
        elif len(cand_precusor) >= 1:           
            for i in cand_precusor:
                line_i = df.loc[i][2:]
                line_i = [str(i) for i in line_i]
                for x in line_i:
                    try:
                        if 'C' in x:                           
                            cand_formula.append(x)
                    except:
                        print('cand_formula has an error')
    if len(cand_formula)==1:
        cand_formula = cand_formula
    if len(cand_formula)>1:
        cand_formula = sorted(list(set(cand_formula)), key = cand_formula.index) 
    return cand_formula

def dict_slice(df,precusor):
    if 0<precusor<=100:
        df = df[:55]
    elif 100<precusor<=200:
        df = df[51:718]
    elif 200<precusor<=300:
        df = df[714:2073]
    elif 300<precusor<=400:
        df = df[2069:3747]
    elif 400<precusor<=500:
        df = df[3743:5043]
    elif 500<precusor<=600:
        df = df[5039:5771]
    elif 600<precusor<=700:
        df = df[5767:6082]
    elif 700<precusor<=800:
        df = df[6078:6342]
    elif 800<precusor<=900:
        df = df[6338:6542]
    elif 900<precusor<=1000:
        df = df[6538:6636]
    elif 1000<precusor:
        df = df[6532:]
    return df

def formula_stand(a):
    e = dict(C=12,Cx=13.003355,H=1.007825,Hx=2.014102,N=14.003074,Nx=15.000109,O=15.994915,Ox=16.999131,Oy=17.999159,P=30.973763,S=31.972072,Sx=32.971459,Sy=33.967868,Cl=34.968853,Cly=36.965903,Br=78.918336,Bry=80.91629,I=126.904477,G=27.976928,Gx=28.976496,Gy=29.973772,F=18.998403,Hu=1.007276,Hd=1.008374)
    if 'Si' in a:
        a = a.replace('Si','G')
    if 'C' in a:
        c = re.findall(r'C(\d\d?)',a)
        if len(c) == 0:
            c = 1
        else:
            c = int(c[0])
    elif 'C' not in a:
        c = 0
    if 'H' in a:
        h = re.findall(r'H(\d\d?)',a)
        if len(h) == 0:
            h = 1
        else:
            h = int(h[0])
    elif 'H' not in a:
        h = 0
    if 'O' in a:
        o = re.findall(r'O(\d\d?)',a)
        if len(o) == 0:
            o = 1
        else:
            o = int(o[0])
    elif 'O' not in a:
        o = 0
    if 'N' in a:
        n = re.findall(r'N(\d\d?)',a)
        if len(n) == 0:
            n = 1
        else:
            n = int(n[0])
    elif 'N' not in a:
        n = 0
    if 'P' in a:
        p = re.findall(r'P(\d\d?)',a)
        if len(p) == 0:
            p = 1
        else:
            p = int(p[0])
    elif 'P' not in a:
        p = 0
    if 'S' in a:
        s = re.findall(r'S(\d\d?)',a)
        if len(s) == 0:
            s = 1
        else:
            s = int(s[0])
    elif 'S' not in a:
        s = 0
    if 'Cl' in a:
        cl = re.findall(r'Cl(\d\d?)',a)
        if len(cl) == 0:
            cl = 1
        else:
            cl = int(cl[0])
    elif 'Cl' not in a:
        cl = 0
    if 'Br' in a:
        br = re.findall(r'Br(\d\d?)',a)
        if len(br) == 0:
            br = 1
        else:
            br = int(br[0])
    elif 'Br' not in a:
        br = 0
    if 'I' in a:
        i = re.findall(r'I(\d\d?)',a)
        if len(i) == 0:
            i = 1
        else:
            i = int(i[0])
    elif 'I' not in a:
        i = 0
    if 'G' in a:
        g = re.findall(r'G(\d\d?)',a)
        if len(g) == 0:
            g = 1
        else:
            g = int(g[0])
    elif 'G' not in a:
        g = 0
    if 'F' in a:
        f = re.findall(r'F(\d\d?)',a)
        if len(f) == 0:
            f = 1
        else:
            f = int(f[0])
    elif 'F' not in a:
        f = 0
    b_standard = 'C%d' % c
    if h > 0:
        b_standard += 'H%d' % h
        h_temp = 1
    elif h == 0:
        h_temp = 0
    if o > 0:
        b_standard += 'O%d' % o
        o_temp = 1
    elif o == 0:
        o_temp = 0  
    if n > 0:
        b_standard += 'N%d' % n
        n_temp = 1
    elif n == 0:
        n_temp = 0
    if p > 0:
        b_standard += 'P%d' % p
        p_temp = 1
    elif p == 0:
        p_temp = 0
    if s > 0:
        b_standard += 'S%d' % s
        s_temp = 1
    elif s == 0:
        s_temp = 0
    if cl > 0:
        b_standard += 'Cl%d' % cl
        cl_temp = 1
    elif cl == 0:
        cl_temp = 0
    if br > 0:
        b_standard += 'Br%d' % br
        br_temp = 1
    elif br == 0:
        br_temp = 0
    if i > 0:
        b_standard += 'I%d' % i
        i_temp = 1
    elif i == 0:
        i_temp = 0
    if g > 0:
        b_standard += 'G%d' % g
        g_temp = 1
    elif g == 0:
        g_temp = 0
    if f > 0:
        b_standard += 'F%d' % f
        f_temp = 1
    elif f== 0:
        f_temp = 0
    precusors = round((e['Hu']+e['C']*c+e['H']*h+e['O']*o+e['N']*n+e['P']*p+e['S']*s+e['Cl']*cl+e['Br']*br+e['I']*i+e['G']*g+e['F']*f),6)
    element_list = torch.tensor([1,1,o_temp,n_temp,p_temp,s_temp,cl_temp,br_temp,i_temp,g_temp,f_temp,o])
    return b_standard, precusors, element_list

correction_rate, correction_first_number_rate, unmatch = match(input_file)
print("correction rate of evalset:{0:.2f}%".format(correction_rate))
print("correction rate of first_match:{0:.2f}%".format(correction_first_number_rate))
print(unmatch)