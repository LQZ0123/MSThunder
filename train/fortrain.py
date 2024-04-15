from calendar import c
from pickle import NONE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from fordata import DataLoader
from tqdm import tqdm
import spacy
import numpy as np
import random
import math
import time
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
                 max_length = 52):
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
    def __init__(self,hid_dim,dropout,max_len = 52):
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
                 max_length = 12):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
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
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).long().to(self.device) 
        
        trg = self.dropout(((self.tok_embedding(trg)) * self.scale) + self.pos_embedding(pos)).to(self.device) 
        
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

INPUT_DIM = 2
OUTPUT_DIM = 31
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
TRG_PAD_IDX = -1

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)
LEARNING_RATE = 0.001

optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

scheduler_explr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99, last_epoch=-1, verbose=False)
lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=30, after_scheduler=scheduler_explr)

class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes=OUTPUT_DIM, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            target1 = targets.clone().cuda() 
            target1 =torch.where(target1==TRG_PAD_IDX,0,1) 
            target1 = target1.view(BATCH_SIZE,targets.size(1),-1).cuda()
            targets = torch.empty(size=(BATCH_SIZE,targets.size(1), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(2, targets.data.view(BATCH_SIZE,targets.size(1),-1), 1. - smoothing)
            
            targets = torch.mul(targets,target1).cuda() 
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


criterion_ce = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
label_loss = LabelSmoothCrossEntropyLoss(smoothing=0.1)

def train(model, iterator, optimizer, criterion_ce, lr_scheduler,clip):    
    model.train()   
    epoch_loss = 0
    
    for i,(src,trg) in tqdm(enumerate(iterator)):
        src = src.permute(0,1,2)
        trg = trg.squeeze(1)
        
        src = src.to(device)
        trg = trg.long().to(device)
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])

        output2 = output.clone().contiguous()
        trg2 = trg[:,1:].clone().contiguous()
        loss = label_loss(output2,trg2)

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
            
            output2 = output.clone().contiguous()
            trg2 = trg[:,1:].clone().contiguous()
            loss = label_loss(output2,trg2)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def load_input(input_path):
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

def translate_sentence(input_pathtest, model, device, max_len):
    
    model.eval()
    trgx = []
    for i, src in tqdm(enumerate(input_pathtest)):
        src = src.unsqueeze(0)
        src = src.to(device)
        
        src_mask = model.make_src_mask(src)

        with torch.no_grad():
            enc_src = model.encoder(src, src_mask)
             
        trg_indexes = [29]

        
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            
            trg_mask = model.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
    
            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)
            if pred_token == 30:
                break
        trgx.append(trg_indexes[1:-1])
       
    return trgx, attention

def target_value(target_path):
    target_total = []
    with open(target_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'compound' in line:
                n = 2
            if n == 1:
                line = line.strip().split(',')
                line = [int(i) for i in line]
                target_total.append(line)
            n -= 1
    return target_total

def correction(pre,trg):
    correction_number = 0
    correction_numbero2 = 0
    element_number = 0
    o_number = 0
    o2_number = 0
    total_number = 0
    try:
        assert len(pre) == len(trg), \
        "pre and trg must have the same len"
    except:
        import ipdb;ipdb.set_trace()
    for i,x in zip(pre,trg):
        total_number +=1
        if i == x:
            correction_number += 1
        if i[:-1] == x[:-1]:
            element_number += 1
        if i[-1] == x[-1]:
            o_number += 1
        if abs(i[-1]-x[-1])<3:
            o2_number += 1
        if abs(i[-1]-x[-1])<3 and i[:-1] == x[:-1]:
            correction_numbero2 += 1
    correction_rate = (correction_number/total_number) * 100
    element_rate = (element_number/total_number) * 100
    o_rate = (o_number/total_number) * 100
    o2_rate = (o2_number/total_number) * 100
    correction_o2_rate = (correction_numbero2/total_number) * 100
    return correction_rate,element_rate,o_rate,o2_rate,correction_o2_rate

best_valid_loss = float('inf')
best_train_loss = float('inf')
best_accuracy = 0

evalset_path = './file/challenge2016_for_MS2_input.txt'
evaltarget_path = './file/challenge2016_for_MS2_output.txt'

data_train = DataLoader("./file/for_input.txt", "./file/for_output.txt", mode = 'train')
data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, pin_memory=True, \
        num_workers=2, drop_last=True, shuffle=True)\

data_train2 = DataLoader(evalset_path,evaltarget_path, mode = 'eval')
data_loader_train2 = torch.utils.data.DataLoader(data_train2, batch_size=64, pin_memory=True, \
        num_workers=2, drop_last=True, shuffle=True)

'tensorboard'
'===================================================================================='
writer = SummaryWriter(log_dir='./challenge2016_accuracy')
'===================================================================================='

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train(model, data_loader_train, optimizer, criterion_ce, lr_scheduler, CLIP)
    valid_loss = evaluate(model, data_loader_train2, criterion_ce, epoch)
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join('./checkpoints/', 'formulapre-model-p1.pt'))
        print("Save best model to" + os.path.join('./checkpoints/', 'formulapre-model-p1.pt'))
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), os.path.join('./checkpoints/', 'formulapre-model-p2.pt'))
        print("Save best model to" + os.path.join('./checkpoints/', 'formulapre-model-p2.pt'))

    'tensorboard'
    '===================================================================================='
    trg_pre_eval1,attention_eval1 = translate_sentence(load_input(evalset_path), model, device, max_len = 12)
    trg_trg_eval1 = target_value(evaltarget_path)
    eval_rate1,_,_,_,_ = correction(trg_pre_eval1,trg_trg_eval1)
    writer.add_scalar(tag="accuracy",
                      scalar_value = eval_rate1, 
                      global_step = epoch+1 
                      )

    '===================================================================================='

    if eval_rate1 >= best_accuracy:
        best_accuracy = eval_rate1
        torch.save(model.state_dict(), os.path.join('./checkpoints/', 'formulapre-model-p3.pt'))
        print("Save best model to" + os.path.join('./checkpoints/', 'formulapre-model-p3.pt'))
    print(epoch+1, optimizer.param_groups[0]['lr'])

model.load_state_dict(torch.load(os.path.join('./checkpoints/', 'formulapre-model-p3.pt')))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

trg_pre_eval,attention_eval = translate_sentence(load_input(evalset_path), model, device, max_len = 12)
trg_trg_eval = target_value(evaltarget_path)
correction_rate,element_rate,o_rate,o2_rate, correction_o2_rate = correction(trg_pre_eval,trg_trg_eval)

print("correction rate of evalset:{0:.2f}%".format(correction_rate))
print("correction rate of element_rate:{0:.2f}%".format(element_rate))
print("correction rate of o_rate:{0:.2f}%".format(o_rate))
print("correction rate of o2_rate:{0:.2f}%".format(o2_rate))
print("correction rate of correction_o2_rate:{0:.2f}%".format(correction_o2_rate))

with open(evalset_path,'r') as f:
    formula_t = []
    for line in f.readlines():
        if 'compound' in line:
            line = line.strip().split(':')
            formula_t.append(line[1]+':'+line[2])

with open('./file/challenge2016-for-p3.txt','w') as f:
    n = 0
    for i,j,x in zip(trg_trg_eval, trg_pre_eval,formula_t):
        n += 1
        f.write('compound_{0}:{1}'.format(n,x) + '\n')
        f.write('t:' + str(i)[1:-1] + '\n')
        f.write('p:' + str(j)[1:-1] + '\n')

