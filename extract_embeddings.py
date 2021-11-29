import scipy.io.wavfile as wav
import pandas as pd
import random
import soundfile
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np


# df_100 = pd.read_csv('/mnt/data0/wc43/prep4kaldi/train-clean-100-spk.txt', names = ['speaker'], header = 0)
# df_360 = pd.read_csv('/mnt/data0/wc43/prep4kaldi/train-clean-360-spk.txt', names = ['speaker'], header = 0)
# df_500 = pd.read_csv('/mnt/data0/wc43/prep4kaldi/train-other-500-spk.txt', names = ['speaker'], header = 0)
# df_all = pd.concat([df_100, df_360, df_500], names = ['speaker'], ignore_index=True)
df_all = pd.read_csv('/mnt/data0/wc43/prep4kaldi/dev-clean-spk.txt', names = ['speaker'], header = 0)
my_dict = {};
for i in range(len(df_all)):
    spk_selected = df_all['speaker'][i]
#     fpath = Path("/mnt/data0/wc43/PVAD_data/LibriSpeech",str(spk_selected))
    fpath = Path("/mnt/data0/wc43/PVAD_dev/LibriSpeech/dev-clean",str(spk_selected))
    wav_fpaths = list(fpath.glob("**/*.wav"))
    wavs = [preprocess_wav(wav_fpaths) for wav_fpaths in wav_fpaths]
    encoder = VoiceEncoder()
    embed = encoder.embed_speaker(wavs)
    my_dict[spk_selected]=embed
# np.save('spk_to_embed.npy', my_dict)
np.save('spk_to_embed_test.npy', my_dict)
