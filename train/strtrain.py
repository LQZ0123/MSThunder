from calendar import c
from pickle import NONE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from strdata import DataLoader
from tqdm import tqdm
import spacy
import numpy as np
import random
import math
import time
import re
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256

class Encoder(nn.Module):
    def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,dropout,device,
                 max_length = 62):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Linear(input_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim,dropout)   
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)]) 
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        src = self.tok_embedding(src)

        src = self.pos_embedding(src.transpose(0,1)).transpose(0,1).to(device)
        
        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class EncoderLayer(nn.Module):
    def __init__(self,hid_dim,n_heads,pf_dim,dropout,device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim) 
        
        self.ff_layer_norm = nn.LayerNorm(hid_dim) 
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,pf_dim,dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        _src = self.positionwise_feedforward(src)
        
        src = self.ff_layer_norm(src + self.dropout(_src))
       
        return src
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim       
        self.n_heads = n_heads        
        self.head_dim = hid_dim // n_heads 
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        Q = self.fc_q(query) 
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10) 
        attention = torch.softmax(energy, dim = -1)
        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        x = self.fc_o(x)
        
        return x, attention
        
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout) #0.1
        
    def forward(self, x):
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        x = self.fc_2(x)
        
        return x 

class PositionalEncoding(nn.Module):
    def __init__(self,hid_dim,dropout,max_len = 132):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len,hid_dim)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2).float() * (-math.log(10000.0) / hid_dim)) 
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe) 

    def forward(self, x):
        
        x = x + self.pe[:x.size(0), :]
        
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self,output_dim,hid_dim,n_layers,n_heads,pf_dim,dropout,device,
                 max_length = 132):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim) 
        self.pos_embedding = PositionalEncoding(hid_dim,dropout) 
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        trg = self.tok_embedding(trg) 
        trg = self.pos_embedding(trg.transpose(0,1)).transpose(0,1).to(device) 
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
        
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self,hid_dim,n_heads,pf_dim,dropout,device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
       
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
       
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        
        _trg = self.positionwise_feedforward(trg)
       
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,src_pad_idx,trg_pad_idx,device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
       
        src_mask = (src[:,:,0] != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        return src_mask
    
    def make_trg_mask(self, trg):
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
         
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return trg_mask
    def forward(self, src, trg):
           
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask) 
          
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask) 
        
        return output, attention 
    
INPUT_DIM = 14
OUTPUT_DIM = 43 
HID_DIM = 256 
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

N_EPOCHS = 300
CLIP = 1

enc = Encoder(INPUT_DIM,HID_DIM,ENC_LAYERS,ENC_HEADS,ENC_PF_DIM,ENC_DROPOUT,device)

dec = Decoder(OUTPUT_DIM,HID_DIM,DEC_LAYERS,DEC_HEADS,DEC_PF_DIM,DEC_DROPOUT,device)

SRC_PAD_IDX = -1
TRG_PAD_IDX = 1

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights)

LEARNING_RATE = 0.002
optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)
scheduler_coslr =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=30, after_scheduler=scheduler_coslr)
criterion_ce = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
cosine_loss = nn.CosineEmbeddingLoss(margin=0.2)

def cos_loss(output, trg):
    target = F.one_hot(trg, OUTPUT_DIM)
    output = output[target==1].cuda()
    output = output.reshape((BATCH_SIZE, 131))
    y = torch.ones((BATCH_SIZE)).cuda()
    trg[trg==TRG_PAD_IDX] = 0
    trg = trg.cuda()
    loss = cosine_loss(output, trg, y)
    return loss

def train(model, iterator, optimizer, lr_scheduler, criterion_ce, clip):
    
    model.train()
    epoch_loss = 0
    
    for i,(src,trg) in tqdm(enumerate(iterator)):
        
        src = src.permute(0,1,2) 
        trg = trg.squeeze(1)  
        
        src = src.to(device)
        trg = trg.long().to(device)
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
       
        output1 = output.clone().contiguous()
        trg1 = trg[:,1:].clone().contiguous()
        loss_cos = cos_loss(output1,trg1)

        output_dim = output.shape[-1]
        
        
        output = output.contiguous().view(-1,output_dim) 
        trg = trg[:,1:].contiguous().view(-1)
         
        loss = criterion_ce(output, trg)
        loss += 0.5*loss_cos
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
       
    lr_scheduler.step()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion_ce, epoch):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, (src,trg) in tqdm(enumerate(iterator)):
            
            src = src.permute(0,1,2)
            
            trg = trg.squeeze(1)
            src = src.to(device)
            trg = trg.long().to(device)
            optimizer.zero_grad()
            output, _ = model(src, trg[:,:-1])
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1,output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            loss = criterion_ce(output, trg)
            
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def load_input(input_path):
        input_list = []
        input_tensor = torch.zeros((62,14))
        idx = 0
        with open(input_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'compound' in line:
                    n = 2
                if n == 1:
                    string = line.strip().split(',')
                if n == 0:
                    intensity = line.strip().split(',')
                    assert len(string) == len(intensity), \
                        "string and intensity must have the same len"
                    
                    for i in range(len(string)):
                        convert_list(string[i],input_tensor,i,intensity[i])  
                    input_list.append(input_tensor)
                    input_tensor = torch.zeros((62,14))                   
                    idx += 1
                n -= 1
        return input_list

def convert_list(string,tensor,id,intensity):
        
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

look_upr = {'0':'e','1':'a','2':'s','3':'H', '4':'C', '5':'N','6':'S','7':'F','8':'O','9':'P','10':'I','11':'X',\
        '12':'Y', '13':'G', '14':'B', '15':'Z', '16':'-', '17':'=', '18':'#', '19':'(', \
        '20':')', '21':'+', '22':'-', '23':'1', '24':'2', '25':'3', '26':'4', '27':'5', \
            '28':'6', '29':'7', '30':'8', '31':'9', '32':'0', '33':'@', '34':'[', '35':']', \
            '36':'/', '37':'\\','38':'%','39':'.','40':'M','41':'T','42':'e'}

def translate_sentence(input_pathtest, model, device, max_len = 132):
  
    model.eval()
    trgx = []
    
    for i, src in tqdm(enumerate(input_pathtest)):
        src = src.unsqueeze(0)
        src = src.to(device)
        
        src_mask = model.make_src_mask(src)
        with torch.no_grad():
            enc_src = model.encoder(src, src_mask)
        
        trg_indexes = [2]
   
        for i in range(max_len):
            
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
           
            trg_mask = model.make_trg_mask(trg_tensor) 

            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            
            pred_token = output.argmax(2)[:,-1].item()
            
            trg_indexes.append(pred_token)
            
            if pred_token == 42:
                break
        trg_temp = ''
        for i in trg_indexes:
            trg_temp += look_upr[str(i)]
        trgx.append(trg_temp[1:-1])
        
    return trgx, attention
    
def target_value(target_path):
    target_total = []
    with open(target_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'compound' in line:
                n = 1
            if n== 0:
                line = line.replace('Cl', 'X')
                line = line.replace('Br', 'Y')
                line = line.replace('Si', 'G')
                line = line.replace('Se', 'Z')
                line = line.replace('10', '0')
                line = line.replace('Na', 'M')
                line = line.replace('Fe','T')
                line = line.replace('K','T')
                line = line.replace('Au','T')
                line = line.replace('As','T')
                line = line.replace('Cr','T')
                line = line.replace('Al','T')
                line = line.strip().upper()
                target_total.append(line)
            n -= 1
    return target_total

def correction(pre,trg):
    correction_number = 0
    total_number = 0
    
    assert len(pre) == len(trg), \
    "pre and trg must have the same len"
    
    for i,j in zip(pre,trg):
        total_number +=1
        if i.strip() == j.strip():
            correction_number += 1
    correction_rate = (correction_number/total_number) * 100
    return correction_rate

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')
best_train_loss = float('inf')
best_accuracy = 0


evalset_path1 = './file/input_file_challenge2016.txt'
evaltarget_path1 = './file/output_file_challenge2016.txt'

evalset_path = './file/str_input_set.txt'
evaltarget_path = './file/str_output_set.txt'

data_train = DataLoader("./file/strinput.txt", "./file/stroutput.txt", mode = 'train')
data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, pin_memory=True, \
        num_workers=2, drop_last=True, shuffle=True)\

data_train2 = DataLoader(evalset_path1, evaltarget_path1, mode = 'eval')
data_loader_train2 = torch.utils.data.DataLoader(data_train2, batch_size=64, pin_memory=True, \
        num_workers=2, drop_last=True, shuffle=True)
    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'tensorboard'
'===================================================================================='
writer = SummaryWriter(log_dir='./challenge2016_accuracy')
'===================================================================================='

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train(model, data_loader_train, optimizer, lr_scheduler, criterion_ce, CLIP)
    valid_loss = evaluate(model, data_loader_train2, criterion_ce, epoch)
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join('./checkpoints/', 'tut-model-p1.pt'))
        print("Save best model to" + os.path.join('./checkpoints/', 'tut-model-p1.pt'))
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), os.path.join('./checkpoints/', 'tut-model-p2.pt'))
        print("Save best model to" + os.path.join('./checkpoints/', 'tut-model-p2.pt'))
    
    'tensorboard'
    '===================================================================================='
    trg_pre_eval1,attention_eval1 = translate_sentence(load_input(evalset_path1), model, device, max_len = 132)
    trg_trg_eval1 = target_value(evaltarget_path1)
    eval_rate1 = correction(trg_pre_eval1,trg_trg_eval1)
    writer.add_scalar(tag="accuracy", 
                      scalar_value = eval_rate1, 
                      global_step = epoch+1  
                      )

    '===================================================================================='
    if eval_rate1 >= best_accuracy:
        best_accuracy = eval_rate1
        torch.save(model.state_dict(), os.path.join('./checkpoints/', 'tut-model-p3.pt'))
        print("Save best model to" + os.path.join('./checkpoints/', 'tut-model-p3.pt'))
    print(epoch+1, optimizer.param_groups[0]['lr'])

model.load_state_dict(torch.load(os.path.join('./checkpoints/', 'tut-model-p3.pt')))

trg_pre_eval,attention_eval = translate_sentence(load_input(evalset_path), model, device, max_len = 132)
trg_trg_eval = target_value(evaltarget_path)
eval_rate = correction(trg_pre_eval,trg_trg_eval)

trg_pre_eval1,attention_eval1 = translate_sentence(load_input(evalset_path1), model, device, max_len = 132)
trg_trg_eval1 = target_value(evaltarget_path1)
eval_rate1 = correction(trg_pre_eval1,trg_trg_eval1)

print("correction rate of evalset:{0:.2f}%".format(eval_rate))
print("correction rate of testset:{0:.2f}%".format(eval_rate1))

def textout(evalset_path,trg_trg_eval,trg_pre_eval,number):
    with open(evalset_path,'r') as f:
        formula_t = []
        for line in f.readlines():
            if 'compound' in line:
                line = line.strip().split(':')
                formula_t.append(line[1])

    with open('{0}_predict_{1}.txt'.format(evalset_path[0:-4],number),'w') as f:
        n = 0
        for i,j,x in zip(trg_trg_eval, trg_pre_eval,formula_t):
            n += 1
            f.write('compound_{0}:{1}'.format(n,x) + '\n')
            f.write('t:' + i + '\n')
            f.write('p:' + j + '\n')
    print('finished')

xxx = textout(evalset_path,trg_trg_eval,trg_pre_eval,'p3')
yyy = textout(evalset_path1,trg_trg_eval1,trg_pre_eval1,'p3')
