from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import scipy.io.wavfile as wav
from audiomentations import AddBackgroundNoise, ApplyImpulseResponse
import glob
import soundfile

SAMPLE_RATE = 16000

augment = Compose([
    ApplyImpulseResponse(ir_path='/dataroot/RIRS_NOISES/simulated_rirs', p=0.2,leave_length_unchanged = True),
    AddBackgroundNoise(sounds_path='/dataroot/TUT-acoustic-scenes-2017-development/30s', min_snr_in_db=3,max_snr_in_db=24,p=1)
])

# for file in glob.glob("/dataroot/PVAD_concat/*.wav"):
for file in glob.glob("/dataroot/PVAD_concat_test/*.wav"):
    data, samplerate = soundfile.read(file)
    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=data, sample_rate=SAMPLE_RATE)
    filename = file.split('/')[-1]
#     wav.write("/dataroot/PVAD_concat_aug/%s"%filename, SAMPLE_RATE, augmented_samples.astype(np.float32))
    wav.write("/dataroot/PVAD_concat_aug_test/%s"%filename, SAMPLE_RATE, augmented_samples.astype(np.float32))