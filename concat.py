# 1. concat utterances-clean and labels
# make sure original utts are n*10ms
import scipy.io.wavfile as wav
import pandas as pd
import random
import soundfile
from pydub import AudioSegment

# concate three dataset
# df_100 = pd.read_csv('/dataroot/prep4kaldi/train-clean-100-spk.txt', names = ['speaker'], header = 0)
# df_360 = pd.read_csv('/dataroot/prep4kaldi/train-clean-360-spk.txt', names = ['speaker'], header = 0)
# df_500 = pd.read_csv('/dataroot/prep4kaldi/train-other-500-spk.txt', names = ['speaker'], header = 0)
# df_all = pd.concat([df_100, df_360, df_500], names = ['speaker'], ignore_index=True)

df_all = pd.read_csv('/dataroot/prep4kaldi/dev-clean-spk.txt', names = ['speaker'], header = 0)

index = 0
# f = open("/dataroot/prep4kaldi/concat_labels.txt", "a")
f = open("/dataroot/prep4kaldi/concat_labels_test.txt", "a")
f.truncate(0)
big_hashset = set()
flag = True
    
while(flag):
    # everytime(index th time) for a potential concat
    concat_audio = AudioSegment.empty()
    concat_labels = ""
    num_of_utt = random.randint(1,3)
    hashset = set()
    spklist = []

    for _ in range(0,num_of_utt):
#         if(len(big_hashset) >= 2337 or len(df_all) < 1):
        if(len(big_hashset) >= 39 or len(df_all) < 1):
            flag = False
            break
        # randomly select a spk
        index_selected_spk = random.randint(0,len(df_all) - 1)
        spk_selected = df_all['speaker'][index_selected_spk]
        spklist.append(spk_selected)

        if spk_selected in hashset:
            continue
        hashset.add(spk_selected)

        # randomly select one utt
#         df_utt = pd.read_csv('/dataroot/prep4kaldi/10ms_labels_copy/' + str(spk_selected) + '_frame_labels.csv', header = None, names = ['file','labels'])
        df_utt = pd.read_csv('/dataroot/prep4kaldi/10ms_labels_test/' + str(spk_selected) + '_frame_labels.csv', header = None, names = ['file','labels'])
        index_selected_utt = random.randint(0,len(df_utt) - 1)

        utt_file = df_utt['file'][index_selected_utt]

        # make sure it's n*10ms
        # labels的长度肯定是正确的 按此来调整audio
        # data, samplerate = soundfile.read(utt_file)
        (rate,sig) = wav.read(utt_file)
        length = sig.shape[0] / rate * 1000

        sound = AudioSegment.from_file(utt_file, format='wav', frame_rate=16000)
        labels = df_utt['labels'][index_selected_utt]

        sound = sound[:len(labels)*10]

        # concat audio
        concat_audio = concat_audio + sound
        # concat labels
        concat_labels = concat_labels + '|' + labels

        df_utt =df_utt.drop([index_selected_utt]).reset_index(drop=True)
#         print('utt deleted')

        # delete speaker if empty
        if(len(df_utt) > 0):
#             df_utt.to_csv('/dataroot/prep4kaldi/10ms_labels_copy/%s_frame_labels.csv'%str(spk_selected),index = False, header = None )
            df_utt.to_csv('/dataroot/prep4kaldi/10ms_labels_test/%s_frame_labels.csv'%str(spk_selected),index = False, header = None )
        else:
            print('deleting ',spk_selected)
            df_all =df_all.drop([index_selected_spk]).reset_index(drop=True)
            big_hashset.add(spk_selected)

#     decide target and store
    target_index = random.randint(0,len(spklist)-1)
#     concat_audio.export('/dataroot/PVAD_concat/%d.wav'%index, format='wav')
    concat_audio.export('/dataroot/PVAD_concat_test/%d.wav'%index, format='wav')
    f.write(str(index) + ',' + str(target_index) + ',' + str(spklist[target_index]) + ',' + concat_labels+'\n')
    index=index+1
    
    
f.close()
print("produced",index,"concat utterances")    
    