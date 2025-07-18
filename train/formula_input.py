import os
from pickle import TRUE
import re
import numpy as np
import random
from tqdm import tqdm

def subformulamatch(sample,item_number,mode):
    e = dict(C=12,Cx=13.003355,H=1.007825,Hx=2.014102,N=14.003074,Nx=15.000109,O=15.994915,Ox=16.999131,Oy=17.999159,P=30.973763,S=31.972072,Sx=32.971459,Sy=33.967868,Cl=34.968853,Cly=36.965903,Br=78.918336,Bry=80.91629,I=126.904477,G=27.976928,Gx=28.976496,Gy=29.973772,F=18.998403,Hu=1.007276,Hd=1.008374)
    input_file = './{0}/subcomp/{1}/{1}_compress_MS2_formulamatch.txt'.format(sample,item_number)
    total = []
    numb_temp = []
    form = []
    mz = []
    intensity = []
    output = []
    number1 = 'compound_1:'
    compound = False
    peak = False
    n_a = 0
    formula_judge = False
    smile = False
    with open (input_file, 'r', encoding = 'UTF-8') as f:
        for line in tqdm(f.readlines()):
            if 'compound' in line:
                number2 = number1
                number1 = line.strip()
            if 'name' in line:
                pass
            if 'precusor' in line:
                precusor_temp = line.strip().split(':')[1]
            if 'formula' in line:
                b = [line.strip().split(':')[1]]
                if 'N/A' in line:
                    formula_judge = False
                elif 'N/A' not in line:
                    formula_judge = True
            if 'smile' in line:
                smile_temp = line.strip().split(':')[1]
                if formula_judge:       
                    for a in b:
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
                        massweight_temp = round((e['C']*c+e['H']*h+e['O']*o+e['N']*n+e['P']*p+e['S']*s+e['Cl']*cl+e['Br']*br+e['I']*i+e['G']*g+e['F']*f),6)
                    b_standard = 'C%d' % c
                    if h > 0:
                        b_standard += 'H%d' % h
                    if o > 0:
                        b_standard += 'O%d' % o
                    if n > 0:
                        b_standard += 'N%d' % n
                    if p > 0:
                        b_standard += 'P%d' % p
                    if s > 0:
                        b_standard += 'S%d' % s
                    if cl > 0:
                        b_standard += 'Cl%d' % cl
                    if br > 0:
                        b_standard += 'Br%d' % br
                    if i > 0:
                        b_standard += 'I%d' % i
                    if g > 0:
                        b_standard += 'G%d' % g
                    if f > 0:
                        b_standard += 'F%d' % f
                    C_number = list(range(1,c+1))
                    Cx_number = list(range(0,3))
                    H_number = list(range(1,h+1))
                    N_number = list(range(0,n+1))
                    O_number = list(range(0,o+1))
                    P_number = list(range(0,p+1))
                    S_number = list(range(0,s+1))
                    Cl_number = list(range(0,cl+1))
                    Cly_number = list(range(0,cl+1))
                    Br_number = list(range(0,br+1))
                    Bry_number = list(range(0,br+1))
                    I_number = list(range(0,i+1))
                    G_number = list(range(0,g+1))
                    F_number = list(range(0,f+1))

                    formula = []
                    massweight = []
                    x = 0
                    for n_i in N_number:
                        for o_i in O_number:
                            for p_i in P_number:
                                for s_i in S_number:                    
                                    for cl_i in Cl_number:
                                        for cly_i in Cly_number:
                                            if cl_i+cly_i > cl:
                                                x += 1
                                                continue
                                            else:
                                                for br_i in Br_number:
                                                    for bry_i in Bry_number:
                                                        if br_i+bry_i > br:
                                                            x += 1
                                                            continue
                                                        else:
                                                            for i_i in I_number:
                                                                for g_i in G_number:
                                                                    for f_i in F_number:
                                                                        for c_i in C_number:
                                                                            for cx_i in Cx_number:
                                                                                if c_i+cx_i > c:
                                                                                    x += 1
                                                                                    continue
                                                                                else:
                                                                                    for h_i in H_number:
                                                                                        if float(h_i)/float(c_i+cx_i) < 0.1 or float(h_i)/float(c_i+cx_i) >3.5:
                                                                                            x += 1
                                                                                            continue
                                                                                        else:
                                                                                            formula_temp = 'C%d' % c_i
                                                                                            if cx_i > 0:
                                                                                                formula_temp += 'Cx%d' % cx_i
                                                                                            if h_i > 0:
                                                                                                formula_temp += 'H%d' % h_i
                                                                                            if o_i > 0:
                                                                                                formula_temp += 'O%d' % o_i
                                                                                            if n_i > 0:
                                                                                                formula_temp += 'N%d' % n_i
                                                                                            if p_i > 0:
                                                                                                formula_temp += 'P%d' % p_i
                                                                                            if s_i > 0:
                                                                                                formula_temp += 'S%d' % s_i
                                                                                            if cl_i > 0:
                                                                                                formula_temp += 'Cl%d' % cl_i
                                                                                            if cly_i > 0:
                                                                                                formula_temp += 'Cly%d' % cly_i
                                                                                            if br_i > 0:
                                                                                                formula_temp += 'Br%d' % br_i
                                                                                            if bry_i > 0:
                                                                                                formula_temp += 'Bry%d' % bry_i
                                                                                            if i_i > 0:
                                                                                                formula_temp += 'I%d' % i_i
                                                                                            if g_i > 0:
                                                                                                formula_temp += 'G%d' % g_i
                                                                                            if f_i > 0:
                                                                                                formula_temp += 'F%d' % f_i
                                                                                            formula.append(formula_temp)
                                                                                            if mode == 'pos': 
                                                                                                massweight_temp2 = (e['Hu']+e['C']*c_i+e['Cx']*cx_i+e['H']*h_i+e['O']*o_i+e['N']*n_i+e['P']*p_i+e['S']*s_i+e['Cl']*cl_i+e['Cly']*cly_i+e['Br']*br_i+e['Bry']*bry_i+e['I']*i_i+e['G']*g_i+e['F']*f_i) 
                                                                                            if mode == 'neg': 
                                                                                                massweight_temp2 = (-e['Hu']+e['C']*c_i+e['Cx']*cx_i+e['H']*h_i+e['O']*o_i+e['N']*n_i+e['P']*p_i+e['S']*s_i+e['Cl']*cl_i+e['Cly']*cly_i+e['Br']*br_i+e['Bry']*bry_i+e['I']*i_i+e['G']*g_i+e['F']*f_i)                                        
                                                                                            
                                                                                            massweight.append(massweight_temp2)  
        
            if 'peak' in line and formula_judge == True:
                peak = True
                continue
            if peak:
                if len(line.split()) == 2:
                    mz_actual = float(line.split()[0])               
                    for xx,element in enumerate(massweight):
                        if abs(((mz_actual-element)*1000000)/element) <= 10:
                            numb_temp.append(xx)
                    form_temp = list(formula[yy] for yy in numb_temp)
                    if len(form_temp) > 0:
                        if len(form_temp) == 1:
                            formula_result = form_temp[0]
                        elif len(form_temp) > 1:
                            formula_minisotope_list = []
                            isotope_number = []                       
                            isotope_number_min = []
                            Ctotal_number = []
                            Ctotal_number_max = []
                            for iii,element1 in enumerate(form_temp):
                                Cxiii = re.findall(r'Cx(\d\d?)',element1)
                                Clyiii = re.findall(r'Cly(\d\d?)',element1)
                                Bryiii = re.findall(r'Bry(\d\d?)',element1)

                                if len(Cxiii) == 0:
                                    Cxiii = 0
                                elif len(Cxiii) == 1:
                                    Cxiii = int(Cxiii[0])

                                if len(Clyiii) == 0:
                                    Clyiii = 0
                                elif len(Clyiii) == 1:
                                    Clyiii = int(Clyiii[0])

                                if len(Bryiii) == 0:
                                    Bryiii = 0
                                elif len(Bryiii) == 1:
                                    Bryiii = int(Bryiii[0])
                                isotope_number_temp = Cxiii + Clyiii + Bryiii
                                isotope_number.append(isotope_number_temp)
                            isotope_number_min_temp = min(isotope_number)
                            for jjj,element2 in enumerate(isotope_number):
                                if element2 == isotope_number_min_temp:
                                    isotope_number_min.append(jjj)

                            for lll in isotope_number_min:
                                formula_minisotope_list.append(form_temp[lll])

                            if len(formula_minisotope_list) == 1:
                                formula_result = formula_minisotope_list[0]
                            if len(formula_minisotope_list) > 1:
                                for mmm, element3 in enumerate(formula_minisotope_list):
                                    Ciii = int(re.findall(r'C(\d\d?)',element3)[0])
                                    Cxiii2 = re.findall(r'Cx(\d\d?)',element3)
                                    if len(Cxiii2) == 0:
                                        Cxiii2 = 0
                                    elif len(Cxiii2) == 1:
                                        Cxiii2 = int(Cxiii2[0])
                                    Ctotal_number.append(Ciii+Cxiii2)
                                Ctotal_number_max_temp = max(Ctotal_number)
                                for kkk,element4 in enumerate(Ctotal_number):
                                    if element4 == Ctotal_number_max_temp:
                                        Ctotal_number_max.append(formula_minisotope_list[kkk])
                                if len(Ctotal_number_max) == 1:
                                    formula_result = Ctotal_number_max[0]
                                elif len(Ctotal_number_max) > 1:
                                    formula_result = random.choice(Ctotal_number_max)                                 
                        form.append(formula_result)
                        mz.append(mz_actual)
                        intensity.append(float(line.strip().split()[1]))
                    numb_temp = []
                    form_temp = []                
                elif len(line.split()) == 1:
                    n_a += 1
                    
                    total.append(number2 + b_standard + '\n')
                    if len(form) > 59:
                        intensity_temp = np.array(intensity)
                        sort_number = sorted(np.argsort(intensity_temp)[-59:])
                        form = [form[ff] for ff in sort_number]
                        mz = [mz[ff] for ff in sort_number]
                        intensity = [intensity[ff] for ff in sort_number]
                    try:
                        form.index(b_standard)
                    except:
                        form.append(b_standard)
                        mz.append(precusor_temp)
                        intensity.append(10.0)
                    for ii,element6 in enumerate(form):
                        if ii + 1 == len(form):
                            total.append(str(element6))
                        else:
                            total.append(str(element6) + ',')
                    total.append('\n')
                    for jj,element7 in enumerate(intensity):
                        if jj + 1 == len(form):
                            total.append(str(element7))
                        else:
                            total.append(str(element7) + ',')
                    total.append('\n')

                    output.append(number2 + b_standard + '\n')
                    output.append(smile_temp + '\n')
                    form = []
                    mz = []
                    intensity = []
                    formula = []
                    massweight = []
                    peak = False
                    formula_judge = False
                    smile = False

        with open('./{0}/subcomp/{1}/{1}_input_file.txt'.format(sample,item_number), "w") as f:
            for i in total:
                f.write(i)
        with open('./{0}/subcomp/{1}/{1}_output_file.txt'.format(sample,item_number), "w") as f:
            for i in output:
                f.write(i)
    print('The task of subformulamatch have been finished')

