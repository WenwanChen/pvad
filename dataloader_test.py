import numpy as np


data_labels = np.load('fbank_labels_test.npy', allow_pickle=True).item()
data_target= np.load('target_embed_to_seek_test.npy', allow_pickle=True).item()
data_emb = np.load('spk_to_embed_test.npy', allow_pickle=True).item()

# X = []
# Y = []

for index in range(1348):
    # - fbank
    fbank_feat = np.load('fbank_test/%d.npy'%index)

    # - fbank labels
    assert fbank_feat.shape[0] == len(data_labels[str(index)]), str(index)
    list1 = list(data_labels[str(index)])
    list2 = list(map(int,list1))
    np.save('nnLabel_test/%d.npy'%index, list2)
    target = int(data_target[str(index)])
    emb = data_emb[target]

    # merge emb with fbank
    newEmbed= [list(emb)]*fbank_feat.shape[0]
    newEmbed = np.array(newEmbed)
    nninput = np.concatenate((fbank_feat, newEmbed), axis=1)
    np.save('nnData_test/%d.npy'%index, nninput)

