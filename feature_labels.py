# calculate mfcc labels for an original utterance
# input labels of wav at 10ms level
# output aligned feature labels
# 可以对每个用户操作一下 保存在一个里面
# 做成object实时计算呢 还是放dataframe里读取

import numpy as np
import sys
target_dict = {}
labels_dict = {}
# file2 = open('/mnt/data0/wc43/prep4kaldi/concat_labels_corected.txt','r')
file2 = open('/mnt/data0/wc43/prep4kaldi/concat_labels_corected_test.txt','r')
Lines = file2.readlines()
for line in Lines:
    target_dict[line.split(',')[0]]=line.split(',')[1]
    label = line.split(',')[2].strip()
    l = len(label)
    feature_labels = ""
#     make sure i + 2 is in range
    for i in range(0, l - 2, 1):
        a = int(label[i])
        b = int(label[i + 1])
        c = int(label[i + 2])
        if(a == 2 or b == 2 or c==2):
            est = 2
        elif (a+b+c > 1.5):
            est = 1
        else:
            est = 0
        feature_labels = feature_labels + str(est)

#     now i should be l-3?
# 直接加一个0 此时0～l-2都有label了 一共l-1个

    feature_labels= feature_labels + "0"
    labels_dict[line.split(',')[0]]=feature_labels
    
file2.close()
# np.save('target_embed_to_seek.npy', target_dict)
# np.save('fbank_labels.npy', labels_dict)
np.save('target_embed_to_seek_test.npy', target_dict)
np.save('fbank_labels_test.npy', labels_dict)
