import torch
from torch.utils import data
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
        return input,label

    def __len__(self):
        return len(self.input_list)

    def aug(self, input, label):
        input = self.aug_shift1(input)
        input = self.aug_mask(input)
        return input, label
    
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
        sub_tensor += shift_tensor/20000
        temp_tensor = torch.zeros((52,2))
        temp_tensor[start_id:end_id,:] = sub_tensor
        input = temp_tensor
        return input
    
    def aug_mask(self, input, p=0.5):    
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
        if sub_tensor.shape[0] >= 5:
            mask_number = torch.randint(start_id,end_id,(3,))
            for i in mask_number:
                input[i,:] = 0
        return input
    
    def load_input(self,input_path):
        input_list = []
        input_tensor = torch.zeros((52,2))
        idx = 0
        with open(input_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'compound' in line:
                    n = 2
                if n == 1:
                    mz = line.strip().split(',')
                if n == 0:
                    intensity = line.strip().split(',')
                    assert len(mz) == len(intensity), \
                        "string and intensity must have the same len"
                    for i,mzi in enumerate(mz):
                        input_tensor[i+1,0] = float(mzi)
                    for i,intensityi in enumerate(intensity):
                        input_tensor[i+1,1] = float(intensityi)
                    input_list.append(input_tensor)
                    input_tensor = torch.zeros((52,2))                   
                    idx += 1
                n -= 1
        return input_list

    def load_label(self,label_path):
        label_list = []
        label_tensor = torch.zeros((1,12))
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'compound' in line:
                    n = 2
                if n == 1:
                    in_list = line.split(',')
                    for i,data in enumerate(in_list):
                        label_tensor[0,i+1] = float(data)
                    label_tensor[0,0] = 29
                    label_tensor[0,11] = 30
                    label_list.append(label_tensor)
                    label_tensor = torch.zeros((1, 12))
                n -= 1
        return label_list

        
