from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
from pathlib import Path
# sig is float 32 -- aug
# my_dict = {};
# fpath = Path("/mnt/data0/wc43/PVAD_concat_aug")
# wav_fpaths = list(fpath.glob("*.wav"))
# for file in wav_fpaths:
#     (rate,sig) = wav.read(file)
#     fbank_feat = np.float32(logfbank(sig,rate,winlen=0.025,winstep=0.01,nfilt=40))
#     my_dict[file.name.split('.')[0]]=fbank_feat
# np.save('fbank.npy', my_dict)


# fpath = Path("/mnt/data0/wc43/PVAD_concat_aug")
fpath = Path("/dataroot/PVAD_concat_aug_test")
wav_fpaths = list(fpath.glob("*.wav"))
for file in wav_fpaths:
    (rate,sig) = wav.read(file)
    fbank_feat = np.float32(logfbank(sig,rate,winlen=0.025,winstep=0.01,nfilt=40))
#     np.save('fbank/%s.npy'%(file.name.split('.')[0]), fbank_feat)
    np.save('fbank_test/%s.npy'%(file.name.split('.')[0]), fbank_feat)