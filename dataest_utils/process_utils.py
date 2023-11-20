import os
import torch
import random

import librosa
import numpy as np
from rich.progress import track
from scipy.io import wavfile
import utils
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch


def load_wav(wav_path):
    return librosa.load(wav_path, sr=None)


def trim_wav(wav, top_db=40):
    return librosa.effects.trim(wav, top_db=top_db)


def normalize_peak(wav, threshold=1.0):
    peak = np.abs(wav).max()
    if peak > threshold:
        wav = 0.98 * wav / peak
    return wav


def resample_wav(wav, sr, target_sr):
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


def save_wav_to_path(wav, save_path, sr):
    wavfile.write(
        save_path,
        sr,
        (wav * np.iinfo(np.int16).max).astype(np.int16)
    )


def load_audio_wav(wav_path, target_sample_rate=44100, skip_loudnorm=False):
    """
    加载音频，必要时候重采样
    """
    if os.path.exists(wav_path) and '.wav' in wav_path:

        wav, sr = load_wav(wav_path)
        wav, _ = trim_wav(wav)
        wav = normalize_peak(wav)

        if target_sample_rate == sr:
            return wav, sr

        wav = resample_wav(wav, sr, target_sample_rate)
        if not skip_loudnorm:
            wav /= np.max(np.abs(wav))

        return wav, target_sample_rate


def process_one(file_path, sampling_rate, hps, f0p='rmvpe'):
    """
    提取mel，f0,uv,bert,vol
    """
    wav, sr = load_audio_wav(file_path, target_sample_rate=hps.data.sampling_rate)

    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)

    f0_predictor = utils.get_f0_predictor(f0p, sampling_rate=sampling_rate, hop_length=hps.hop_length, device=None,
                                          threshold=0.05)
    f0, uv = f0_predictor.compute_f0_uv(wav)

    if sr != hps.data.sampling_rate:
        raise ValueError(
            "{} SR doesn't match target {} SR".format(
                sr, hps.data.sampling_rate
            )
        )

    assert hps.model.vol_embedding, "vol_embedding must be True, now is hps.model.vol_embedding "
    volume_extractor = utils.Volume_Extractor(hps.hop_length)
    volume = volume_extractor.extract(audio_norm).float()
    return f0, uv, volume
