import os
import random

import numpy as np
import torch
import torch.utils.data

import utils
from modules.mel_processing import spectrogram_torch
from utils import load_wav_to_torch
from dataest_utils import process_utils

"""Multi speaker version"""


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        # wav_file|soft_file|spk_name
        filepaths_and_text = [line.strip().split(split) for line in f]
        random.shuffle(filepaths_and_text)
    print(f"--load file done, lines:{len(filepaths_and_text)}, file: {filename}")
    return filepaths_and_text


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, speaker_name_file, hparams, vol_aug: bool = True):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.unit_interpolate_mode = hparams.data.unit_interpolate_mode
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen
        self.spk_map = dict([(sn.replace("\n", ""), i) for i, sn in enumerate(open(speaker_name_file).readlines())])
        self.vol_emb = hparams.model.vol_embedding
        self.vol_aug = hparams.train.vol_aug and vol_aug
        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_audio(self, filename, c_file, spk):
        audio_data, sampling_rate = process_utils.resample_audio(filename, target_sample_rate=self.sampling_rate)
        audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
        audio = torch.FloatTensor(audio_data.astype(np.float32))

        # audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "Sample Rate not match. Expect {} but got {} from {}".format(
                    self.sampling_rate, sampling_rate, filename))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")

        spec = spectrogram_torch(audio_norm, self.filter_length,
                                 self.sampling_rate, self.hop_length, self.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_filename)

        spk = torch.LongTensor([self.spk_map[spk]])

        f0, uv, volume = process_utils.process_one(filename, hps=self.hparams, f0p='rmvpe',
                                                   sampling_rate=self.sampling_rate)

        f0 = torch.FloatTensor(np.array(f0, dtype=float))
        uv = torch.FloatTensor(np.array(uv, dtype=float))

        assert os.path.exists(c_file), c_file + "  is not exist!!!"
        c = torch.load(c_file)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0], mode=self.unit_interpolate_mode)
        if not self.vol_emb:
            volume = None

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), f0.shape, filename)
        assert abs(audio_norm.shape[1] - lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio_norm = audio_norm[:, :lmin * self.hop_length]
        if volume is not None:
            volume = volume[:lmin]
        return c, f0, spec, audio_norm, spk, uv, volume

    def random_slice(self, c, f0, spec, audio_norm, spk, uv, volume):
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None

        if random.choice([True, False]) and self.vol_aug and volume is not None:
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            audio_norm = audio_norm * (10 ** log10_vol_shift)
            volume = volume * (10 ** log10_vol_shift)
            spec = spectrogram_torch(audio_norm,
                                     self.hparams.data.filter_length,
                                     self.hparams.data.sampling_rate,
                                     self.hparams.data.hop_length,
                                     self.hparams.data.win_length,
                                     center=False)[0]

        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1] - 800)
            end = start + 790
            spec, c, f0, uv = spec[:, start:end], c[:, start:end], f0[start:end], uv[start:end]
            audio_norm = audio_norm[:, start * self.hop_length: end * self.hop_length]
            if volume is not None:
                volume = volume[start:end]
        return c, f0, spec, audio_norm, spk, uv, volume

    def __getitem__(self, index):
        file, c_file, spk = self.audiopaths[index]
        return self.random_slice(*self.get_audio(file, c_file, spk))

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        max_c_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        lengths = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        f0_padded = torch.FloatTensor(len(batch), max_c_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spkids = torch.LongTensor(len(batch), 1)
        uv_padded = torch.FloatTensor(len(batch), max_c_len)
        volume_padded = torch.FloatTensor(len(batch), max_c_len)

        c_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()
        volume_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, :c.size(1)] = c
            lengths[i] = c.size(1)

            f0 = row[1]
            f0_padded[i, :f0.size(0)] = f0

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav

            spkids[i, 0] = row[4]

            uv = row[5]
            uv_padded[i, :uv.size(0)] = uv
            volume = row[6]
            if volume is not None:
                volume_padded[i, :volume.size(0)] = volume
            else:
                volume_padded = None
        return c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded, volume_padded
