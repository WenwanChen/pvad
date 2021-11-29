# 4.correct labels
# file1 = open('/mnt/data0/wc43/prep4kaldi/concat_labels.txt', 'r')
# file2 = open('/mnt/data0/wc43/prep4kaldi/concat_labels_corected.txt','a')
file1 = open('/mnt/data0/wc43/prep4kaldi/concat_labels_test.txt', 'r')
file2 = open('/mnt/data0/wc43/prep4kaldi/concat_labels_corected_test.txt','a')
file2.truncate(0)
Lines = file1.readlines()
for line in Lines:
    new_labels = ""
    labels = line.split(',')[3]
    index = line.split(',')[1]
    num_utt = len(labels.split('|')) - 1
    for i in range(num_utt):
        str_tmp = labels.split('|')[i+1]
        if(i != int(index)):
            str_tmp = labels.split('|')[i+1].replace('1','2')
        new_labels = new_labels + str_tmp
    
    file2.write(line.split(',')[0] + ',' +line.split(',')[2] + ',' + new_labels)
file1.close()
file2.close()

