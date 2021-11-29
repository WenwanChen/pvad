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
from torch.nn.functional import one_hot
from sklearn.metrics import average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
BATCH_SIZE = 64

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
        return 130048
#         return 1024

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = np.load('nnData/%d.npy'%(index + 1))
        y = np.load('nnLabel/%d.npy'%(index + 1))
        return X, y
    
class Dataset_val(torch.utils.data.Dataset):
#     def __init__(self, x, y):
#         'Initialization'
#         self.x = x
#         self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return 9984
#         return 256

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = np.load('nnData/%d.npy'%(index + 130049))
        y = np.load('nnLabel/%d.npy'%(index + 130049))
        return X, y

class WPL(nn.Module):
    """Weighted pairwise loss implementation for three classes.
    The weight pairs are interpreted as follows:
    [<ns,tss> ; <ntss,ns> ; <tss,ntss>]
    Target labels contain indices, the model output is a tensor of probabilites for each class.
    (ns, ntss, tss) -> {0, 1, 2} 
    reference: https://github.com/pirxus/personalVAD/blob/master/src/personal_vad.py 
    """

    def __init__(self, weights=torch.tensor([0.1, 1.0, 1.0])):
        """Initialize the WPL class.
        Args:
            weights (torch.tensor, optional): The weight values for each class pair.
        """

        super(WPL, self).__init__()
        self.weights = weights
        assert len(weights) == 3, "The wpl is defined for three classes only."

    def forward(self, output, target):
        """Compute the WPL for a sequence.
        Args:
            output (torch.tensor): A tensor containing the model predictions.
            target (torch.tensor): A 1D tensor containing the indices of the target classes.
        Returns:
            torch.tensor: A tensor containing the WPL value for the processed sequence.
        """

        output = torch.exp(output)
        label_mask = one_hot(target,3) > 0.5 # boolean mask
        label_mask_r1 = torch.roll(label_mask, 1, 1) # if ntss, then tss
        label_mask_r2 = torch.roll(label_mask, 2, 1) # if ntss, then ns

        # get the probability of the actual labels and the other two into one array
        actual = torch.masked_select(output, label_mask).cuda()
        plus_one = torch.masked_select(output, label_mask_r1).cuda()
        minus_one = torch.masked_select(output, label_mask_r2).cuda()

        # arrays of the first pair weight and the second pair weight used in the equation
        w1 = torch.masked_select(self.weights, label_mask).cuda() # if ntss, w1 is <ntss, ns>
        w2 = torch.masked_select(self.weights, label_mask_r1).cuda() # if ntss, w2 is <tss, ntss>

        # first pair
        first_pair = w1 * torch.log(actual / (actual + minus_one))
        second_pair = w2 * torch.log(actual / (actual + plus_one))

        # get the negative mean value for the two pairs
        wpl = -0.5 * (first_pair + second_pair)

        # sum and average for minibatch
        return torch.mean(wpl)
    
    

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
#         print("self.batchsize", self.batch_size)
#         是因为俩gpu吗。。
        h_0 = torch.randn(2, self.batch_size/2, self.hidden_dim)
        c_0 = torch.randn(2, self.batch_size/2, self.hidden_dim)

        # The Variable API is now semi-deprecated, so we use nn.Parameter instead.
        # Note: For Variable API requires_grad=False by default;
        # For Parameter API requires_grad=True by default.
        h_0 = nn.Parameter(h_0, requires_grad=True)
        c_0 = nn.Parameter(c_0, requires_grad=True)
        return (h_0, c_0)
    
    
    def forward(self, x, x_lengths, y):
        batch_size, seq_len, _ = x.size()      
        x_pack = PACK(x, x_lengths, batch_first=True)
        lstm_out, _ = self.lstm(x_pack.float())
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, \
                                                             total_length = seq_len,padding_value = 3 )

        tag_scores = self.hidden2tag(lstm_out)
        
        return tag_scores
  
        
params = {'batch_size': BATCH_SIZE,
          'shuffle': True}

training_set = Dataset()
training_generator = torch.utils.data.DataLoader(training_set, **params, collate_fn=PadSequence())

val_set = Dataset_val()
val_generator = torch.utils.data.DataLoader(val_set, **params, collate_fn=PadSequence())

print("********** Data loaded *************")
# *****
# gpu_ids = [3,4]
# cuda='cuda:'+ str(gpu_ids[0])
model = MyModel(296, 64, 3, BATCH_SIZE)

# device = torch.device("cuda:3,cuda:4" if torch.cuda.is_available() else "cpu") 

# model = MyModel(296, 64, 3, 8)

# model= nn.DataParallel(model,device_ids = [3,4])
# model.to(device)

# model = MyModel(296, 64, 3, 8).cuda('cuda:3,4')
# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# decayRate = 0.8
# my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0 = 8, T_mult=2, eta_min=0.00005)  # 24_coswarm
my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0 = 8, T_mult=2, eta_min=0.00005)    # 24_coswarm_wpl
model = nn.DataParallel(model).cuda()

# loss_fn = nn.CrossEntropyLoss(ignore_index=3, reduction='mean')
loss_fn = WPL()

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
        
def train(model, train_loader, val_loader, loss_fn, optimizer, n_epochs = 5, batch_size = BATCH_SIZE):
    softmax = nn.Softmax(dim=1)
    epoch_val_losses = []
    epoch_train_losses = []
    for epoch in range(n_epochs):
        train_losses = []
        val_losses = []
        count = 0
        for X_batch, X_lens_batch, y_batch in train_loader:
            optimizer.zero_grad()
            ypred_batch = model(X_batch.cuda(), X_lens_batch.cuda(), y_batch.cuda()).cuda()
            seq_len = len(y_batch[0])
            y_batch = torch.tensor(y_batch, dtype=torch.long).cuda()
            loss = 0
            for j in range(batch_size):
                loss += loss_fn(ypred_batch[j][:X_lens_batch[j]], y_batch[j][:X_lens_batch[j]])

            loss /= batch_size # normalize loss for each batch..
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
            count+=1
            print_percent_done(count, 130048/batch_size)

        my_lr_scheduler.step()
        
        with torch.no_grad():
            print("validating...")
            n_correct = 0
            n_samples = 0
            targets = []
            outputs = []
            for X_val, X_lens_val, y_val in val_loader:
                seq_len = len(y_val[0])
                ypred_val = model(X_val.cuda(), X_lens_val.cuda(), y_val.cuda())
                y_val = torch.tensor(y_val, dtype=torch.long).cuda()
                val_loss = 0
                for j in range(ypred_val.size(0)):
                    classes = torch.argmax(ypred_val[j][:X_lens_val[j]],dim = 1)
                    n_samples += X_lens_val[j]
                    n_correct += torch.sum(classes == y_val[j][:X_lens_val[j]]).item()
                    p = softmax(ypred_val[j][:X_lens_val[j]])
                    outputs.append(p.cpu().numpy())
                    targets.append(y_val[j][:X_lens_val[j]].cpu().numpy())
                    val_loss += loss_fn(ypred_val[j][:X_lens_val[j]], y_val[j][:X_lens_val[j]])
                val_loss /= ypred_val.size(0) # normalize loss for each batch..
            acc = 100.0 * n_correct / n_samples
            val_losses.append(val_loss.item())
            targets = np.concatenate(targets)
            outputs = np.concatenate(outputs)
            targets_oh = np.eye(3)[targets]
            out_AP = average_precision_score(targets_oh, outputs, average=None)
            mAP = average_precision_score(targets_oh, outputs, average='micro')
            print(f"accuracy = {acc:.2f}")
            print(out_AP)
            print(f"mAP: {mAP}")
               
        
        
        print("training loss:                  ", np.mean(train_losses))
        epoch_train_losses.append(np.mean(train_losses))
        print("validation loss:                ", np.mean(val_losses))
        epoch_val_losses.append(np.mean(val_losses))
        state = {
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
        }
        savepath='checkpoint_oct22_coswarm.t7'
        torch.save(state,savepath)

    return epoch_train_losses, epoch_val_losses      

train_loss, val_loss = train(model, training_generator, val_generator, loss_fn, optimizer, n_epochs = 48, batch_size = BATCH_SIZE)



def plot_loss(train_loss, val_loss):
    ''''
    Visualize training loss vs. validation loss.
    Parameters
    ----------
    train_loss: training loss
    val_loss: validation loss
    Returns: None
    -------
    '''
    loss_csv = pd.DataFrame({"iter": range(len(train_loss)), "train_loss": train_loss,
                             "val_loss": val_loss})
    loss_csv.to_csv("loss_oct22_coswarm.csv")
    # gca stands for 'get current axis'
    ax = plt.gca()
    loss_csv.plot(kind='line',x='iter',y='train_loss',ax=ax )
    loss_csv.plot(kind='line',x='iter',y='val_loss', color='red', ax=ax)
#     plt.show()
    plt.savefig("train_vs_val_loss_oct22_coswarm.png")


# # Examine training results
plot_loss(train_loss, val_loss)

