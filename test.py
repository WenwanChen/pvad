import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence as PACK
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
import time
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

BATCH_SIZE = 1

class PadSequence:
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [torch.from_numpy(x[0]) for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True,padding_value = 3)
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = [torch.from_numpy(x[1]) for x in sorted_batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,padding_value = 3)
        return sequences_padded, lengths, labels_padded
    
# defining the Dataset class
class Dataset(torch.utils.data.Dataset):
#     def __init__(self, x, y):
#         'Initialization'
#         self.x = x
#         self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return 1280

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = np.load('nnData_test/%d.npy'%(index + 1))
        y = np.load('nnLabel_test/%d.npy'%(index + 1))
        return X, y
    
    
    

class MyModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size, batch_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2, batch_first=True)  # Note that "batch_first" is set to "True"
        self.hidden2tag = nn.Linear(hidden_dim,output_size)

    def init_hidden(self):
        '''
        Initiate hidden states.
        '''
        # Shape for hidden state and cell state: num_layers * num_directions, batch, hidden_size
        h_0 = torch.randn(2, self.batch_size, self.hidden_dim)
        c_0 = torch.randn(2, self.batch_size, self.hidden_dim)

        # The Variable API is now semi-deprecated, so we use nn.Parameter instead.
        # Note: For Variable API requires_grad=False by default;
        # For Parameter API requires_grad=True by default.
        h_0 = nn.Parameter(h_0, requires_grad=True)
        c_0 = nn.Parameter(c_0, requires_grad=True)
        return (h_0, c_0)
    
    
    def forward(self, x, x_lengths, y):
        self.lstm.flatten_parameters()
        batch_size, seq_len, _ = x.size()      
        x_pack = PACK(x, x_lengths, batch_first=True)
        lstm_out, _ = self.lstm(x_pack.float())
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True,padding_value = 3, \
                                                             total_length = seq_len )
#         print("seq_len:",seq_len)
        tag_scores = self.hidden2tag(lstm_out.reshape(batch_size * seq_len, -1))
        
        return tag_scores
  
        
params = {'batch_size': BATCH_SIZE,
          'shuffle': True}

training_set = Dataset()
training_generator = torch.utils.data.DataLoader(training_set, **params, collate_fn=PadSequence())


print("********** Data loaded *************")

model = MyModel(296, 64, 3, BATCH_SIZE)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
decayRate = 0.9
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# model = nn.DataParallel(model).cuda()

loss_fn = nn.CrossEntropyLoss(ignore_index=3, reduction='mean')

def print_percent_done(index, total, bar_len=80, title='Please wait'):
    '''
    index is expected to be 0 based index. 
    0 <= index < total
    '''
    percent_done = (index+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done', end='\r')

    if round(percent_done) == 100:
        print('\t✅')
        

checkpoint = torch.load('checkpoint_oct22_coswarm.t7')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']


model = nn.DataParallel(model).cuda() #ignore the line if you want to load on Single GPU



def test(model, val_loader, loss_fn, optimizer, batch_size):
    '''
    Inference function
    Parameters
    ----------
    model: LSTM tagger model
    X_test: Test data
    X_lengths_test: Original Test data
    Returns post-softmax probability of individual classes
    postprocess padding classification into 0.
    -------
    '''
    tag_class_acc = torch.tensor([]).cuda()
    y_val_acc = torch.tensor([]).cuda()
    matrix = 0
    count = 0
    with torch.no_grad():
        for X_val, X_lens_val, y_val in val_loader:
            seq_len = len(y_val[0])
            ypred_val = model(X_val.cuda(), X_lens_val.cuda(), y_val.cuda())
            tag_prob = F.softmax(ypred_val, dim = 1).cuda()
            tag_class = torch.argmax(tag_prob, dim=1).cuda()
            count+=1
            if count%10 == 0:
                matrix = matrix + confusion_matrix(y_val_acc.cpu(), tag_class_acc.cpu())
                tag_class_acc = torch.tensor([]).cuda()
                y_val_acc = torch.tensor([]).cuda()
            else:
                y_val_acc = torch.cat((y_val_acc, y_val.view(-1).cuda()), 0)
                tag_class_acc = torch.cat((tag_class_acc, tag_class.view(-1)), 0)
                        
            print_percent_done(count,1280/batch_size)

    return matrix
    


cm = test(model, training_generator, loss_fn, optimizer, BATCH_SIZE)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=["ns","tss","ntss"], yticklabels=["ns","tss","ntss"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
plt.savefig("checkpoint_oct22_coswarm.png")