import os
import sys
import librosa
import numpy as np
from scipy.io.wavfile import read

pdj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pdj)

from process_utils import resample_audio, save_wav_to_path

full_path = '/mnt/cephfs/hjh/train_record/vc/so-vits-svc/vctk/test_data/debug_data/p303/p303_068_mic1.wav'

print("=" * 100)
wav, sr = resample_audio(full_path, target_sample_rate=44100)
rs_wav = (wav * np.iinfo(np.int16).max).astype(np.int16)
print(f"----------wav:{rs_wav}")

save_wav_to_path(wav, save_path="/tmp/test.wav", sr=44100)

print("-" * 100)
sampling_rate, data = read("/tmp/test.wav")
data = data.astype(np.float32)
print(f"-----data:{data}")

wav, sr = librosa.load("/tmp/test.wav", sr=44100)
print("------wav:", wav)
print("------wav:", (wav * np.iinfo(np.int16).max).astype(np.int16).astype(np.float32))
