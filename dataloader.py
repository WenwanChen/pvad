import numpy as np


data_labels = np.load('fbank_labels.npy', allow_pickle=True).item()
data_target= np.load('target_embed_to_seek.npy', allow_pickle=True).item()
data_emb = np.load('spk_to_embed.npy', allow_pickle=True).item()

# X = []
# Y = []

for index in range(140225):
    # - fbank
    fbank_feat = np.load('fbank/%d.npy'%index)
#     (rate,sig) = wav.read("/mnt/data0/wc43/PVAD_concat_aug/%d.wav"%index)
#     fbank_feat = np.float32(logfbank(sig,rate,winlen=0.025,winstep=0.01,nfilt=40))
#     print("fbank_feat.shape:",fbank_feat.shape)

    # - fbank labels
#     print("len of fbank labels",len(data_labels[str(index)]))
    assert fbank_feat.shape[0] == len(data_labels[str(index)]), str(index)
    list1 = list(data_labels[str(index)])
    list2 = list(map(int,list1))
    np.save('nnLabel/%d.npy'%index, list2)
#     int
#     Y.append(list2)
    # - embed
    target = int(data_target[str(index)])
#     print("target spk is:",target)
    emb = data_emb[target]
#     print("embedding shape:",emb.shape) 

    # merge emb with fbank
    newEmbed= [list(emb)]*fbank_feat.shape[0]
    newEmbed = np.array(newEmbed)
#     print("newEmbed.shape:", newEmbed.shape)
    nninput = np.concatenate((fbank_feat, newEmbed), axis=1)
    np.save('nnData/%d.npy'%index, nninput)
    
#     X.append(nninput)
#     print("nninput dim:",nninput.shape)

    if index%10000 == 0 and index > 0:
        print("processed" + str(index) + "data")
#         np.save('X_%d.npy'%index, X)
#         np.save('Y_%d.npy'%index, Y)
#         X = []
#         Y = []