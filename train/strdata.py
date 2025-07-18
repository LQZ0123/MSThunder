import torch
from torch.utils import data
import re
import os
import random

class DataLoader(data.Dataset):
    def __init__(self, input_path, label_path, mode='train'):
        super(DataLoader, self).__init__()
        self.input_list = self.load_input(input_path)
        self.label_list = self.load_label(label_path)
        self.mode = mode

    def __getitem__(self, index):
        input = self.input_list[index]
        label = self.label_list[index]
        if self.mode == 'train':
            input, label = self.aug(input, label)
        return input, label

    def __len__(self):
        return len(self.input_list)
    
    def aug(self, input, label):
        
        input = self.aug_scale(input)
        input = self.aug_shift1(input)
        return input, label
    
    def aug_scale(self, input, p=0.5):
        if random.random() > p:
            return input
        scale = 1.0 + random.uniform(-0.1,0.1)
        return input * scale

    def aug_shift1(self, input, p=0.5):
        if random.random() > p:
            return input
        start_id = 1
        end_id = 1
        for i in range(1, input.shape[0]):
            if input[i][0] == 0:
                end_id = i
                break
        if start_id == end_id:
            return input
        sub_tensor = input[start_id:end_id, :]
        shift_tensor = torch.randint_like(sub_tensor, low=-200, high=200).to(float)
        sub_tensor += shift_tensor/10000
        temp_tensor = torch.zeros((62,14))
        temp_tensor[start_id:end_id,:] = sub_tensor
        input = temp_tensor
        return input

    def load_input(self,input_path):
        input_list = []
        input_tensor = torch.zeros((62,14))
        idx = 0
        with open(input_path,'r') as f:
            for line in f.readlines():
                if 'compound' in line:
                    n = 2
                if n == 1:
                    string = line.strip().split(',')
                if n == 0:
                    intensity = line.strip().split(',')
                    assert len(string) == len(intensity), \
                        "string and intensity must have the same len"
                    for i in range(len(string)):
                        self.convert_list(string[i],input_tensor,i,intensity[i])  
                    input_list.append(input_tensor)
                    input_tensor = torch.zeros((62,14))                   
                    idx += 1
                n -= 1
        return input_list

    def load_label(self,label_path):
        label_list = []
        label_tensor = torch.ones((1,132))
        with open(label_path, 'r') as f:
            for line in f.readlines():
                if 'compound' in line:
                    n = 1
                if n == 0:
                    smile = line.strip()
                    smile_list = self.convert_smile(smile)
                    for i,data in enumerate(smile_list):
                        label_tensor[0,i] = float(data)               
                    label_list.append(label_tensor)
                    label_tensor = torch.zeros((1, 132))
                n -= 1
        return label_list

    def convert_list(self,string,tensor,id,intensity):
        if 'C' in string:
            c_c = re.findall(r'C(\d\d?)',string)
            if len(c_c) == 0:
                c_c = 1
            else:
                c_c = int(c_c[0])
        elif 'C' not in string:
            c_c = 0
        if 'Cx' in string:
            cx_c = re.findall(r'Cx(\d\d?)',string)
            if len(cx_c) == 0:
                cx_c = 1
            else:
                cx_c = int(cx_c[0])
        elif 'Cx' not in string:
            cx_c = 0
        if 'H' in string:
            h_c = re.findall(r'H(\d\d?)',string)
            if len(h_c) == 0:
                h_c = 1
            else:
                h_c = int(h_c[0])
        elif 'H' not in string:
            h_c = 0
        if 'O' in string:
            o_c = re.findall(r'O(\d\d?)',string)
            if len(o_c) == 0:
                o_c = 1
            else:
                o_c = int(o_c[0])
        elif 'O' not in string:
            o_c = 0
        if 'N' in string:
            n_c = re.findall(r'N(\d\d?)',string)
            if len(n_c) == 0:
                n_c = 1
            else:
                n_c = int(n_c[0])
        elif 'N' not in string:
            n_c = 0
        if 'P' in string:
            p_c = re.findall(r'P(\d\d?)',string)
            if len(p_c) == 0:
                p_c = 1
            else:
                p_c = int(p_c[0])
        elif 'P' not in string:
            p_c = 0
        if 'S' in string:
            s_c = re.findall(r'S(\d\d?)',string)
            if len(s_c) == 0:
                s_c = 1
            else:
                s_c = int(s_c[0])
        elif 'S' not in string:
            s_c = 0
        if 'Cl' in string:
            cl_c = re.findall(r'Cl(\d\d?)',string)
            if len(cl_c) == 0:
                cl_c = 1
            else:
                cl_c = int(cl_c[0])
        elif 'Cl' not in string:
            cl_c = 0
        if 'Cly' in string:
            cly_c = re.findall(r'Cly(\d\d?)',string)
            if len(cly_c) == 0:
                cly_c = 1
            else:
                cly_c = int(cly_c[0])
        elif 'Cly' not in string:
            cly_c = 0
        if 'Br' in string:
            br_c = re.findall(r'Br(\d\d?)',string)
            if len(br_c) == 0:
                br_c = 1
            else:
                br_c = int(br_c[0])
        elif 'Br' not in string:
            br_c = 0
        if 'Bry' in string:
            bry_c = re.findall(r'Bry(\d\d?)',string)
            if len(bry_c) == 0:
                bry_c = 1
            else:
                bry_c = int(bry_c[0])
        elif 'Bry' not in string:
            bry_c = 0
        if 'I' in string:
            i_c = re.findall(r'I(\d\d?)',string)
            if len(i_c) == 0:
                i_c = 1
            else:
                i_c = int(i_c[0])
        elif 'I' not in string:
            i_c = 0
        if 'G' in string:
            g_c = re.findall(r'G(\d\d?)',string)
            if len(g_c) == 0:
                g_c = 1
            else:
                g_c = int(g_c[0])
        elif 'G' not in string:
            g_c = 0
        if 'F' in string:
            f_c = re.findall(r'F(\d\d?)',string)
            if len(f_c) == 0:
                f_c = 1
            else:
                f_c = int(f_c[0])
        elif 'F' not in string:
            f_c = 0
        sum_convert = [c_c,cx_c,h_c,n_c,o_c,p_c,s_c,cl_c,cly_c,br_c,bry_c,i_c,g_c,f_c]
        for con,con_ele in enumerate(sum_convert):
            tensor[id+1,con] = con_ele * float(intensity) * 0.01
        
    def convert_smile(self,smile):
        look_up = {'H':'3', 'C':'4', 'N':'5','S':'6','F':'7','O':'8','P':'9','I':'10','X':'11',\
        'Y':'12', 'G':'13', 'B':'14', 'Z':'15', '-':'16', '=':'17', '#':'18', '(':'19', \
        ')':'20', '+':'21', '-':'22', '1':'23', '2':'24', '3':'25', '4':'26', '5':'27', \
            '6':'28', '7':'29', '8':'30', '9':'31', '0':'32', '@':'33', '[':'34', ']':'35', \
            '/':'36', '\\':'37','%':'38','.':'39','M':'40','T':'41'}
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
        smile = smile.strip().upper()
        smile_list = ['1']*132
        smile_list[0] = '2'
        for i in range(len(smile)):
            try:
                smile_list[i+1] = look_up[smile[i]]
            except:
                print(len(smile))
                import ipdb;ipdb.set_trace()            
        smile_list[len(smile)+1] = '42'
        return smile_list

