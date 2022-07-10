from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
import pickle
import random
import os
import tqdm
import sys
import argparse
from pathlib import Path
import julius
from torch import distributed
import torchaudio as ta
from collections import OrderedDict
import math
import torch
from torch.nn import functional as F
import hashlib
import json
sys.path.append('..')


MIXTURE = "mixture"
EXT = ".flac"

import math
import os
import pathlib
import random
import torch


class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, audio_data):
        if random.random() < 0.9:
            random_noise_file = random.choice(self.noise_files_list)

            effects = [
                ['remix', '1'],  # convert to mono
                ['rate', str(self.sample_rate)],  # resample
            ]
            noise, _ = ta.load(str(random_noise_file), normalize=True)
            #print(noise.shape)
            #noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
            audio_length = audio_data.shape[-1]
            noise_length = noise.shape[-1]
            if noise_length > audio_length:
                offset = random.randint(0, noise_length - audio_length)
                noise = noise[..., offset:offset + audio_length]
            elif noise_length < audio_length:
                noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim=-1)
            noise = noise * (0.5+random.random())

            #snr_db = random.randint(self.min_snr_db, self.max_snr_db)
            #snr = math.exp(snr_db / 10)
            #audio_power = audio_data.norm(p=2)
            #noise_power = noise.norm(p=2)
            #scale = snr * noise_power / audio_power
        else:
            noise = (torch.rand(audio_data.shape)*2-1) * 0.1

        return audio_data + noise * random.random()

import torchaudio



class Wavset:
    def __init__(
            self,
            root,
            segment=2, normalize=True,
            samplerate=16000, channels=1, ext=EXT):
        """
        Waveset (or mp3 set for that matter). Can be used to train
        with arbitrary sources. Each track should be one folder inside of `path`.
        The folder should contain files named `{source}.{ext}`.
        Args:
            root (Path or str): root folder for the dataset.
            segment (None or float): segment length in seconds. If `None`, returns entire tracks.
            normalize (bool): normalizes input audio, **based on the metadata content**,
                i.e. the entire track is normalized, not individual extracts.
            samplerate (int): target sample rate. if the file sample rate
                is different, it will be resampled on the fly.
            channels (int): target nb of channels. if different, will be
                changed onthe fly.
            ext (str): extension for audio files (default is .wav).
        samplerate and channels are converted on the fly.
        """
        self.root = Path(root)
        self.segment = segment
        self.normalize = normalize
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        self.num_examples = []
        self.shift = 0

        self.wav_list = glob.glob(f'{self.root}/*/*.flac')
        self.folders_list = glob.glob(f'{self.root}/*')

        self.folder2wav = {}

        for folder in tqdm.tqdm(self.folders_list):
            folder_imgs = glob.glob(f'{folder}/*')
            self.folder2wav[folder] = folder_imgs

        self.N = len(self.wav_list)

        self.transforms_base = transforms.Compose([
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.compose_transform = transforms.Compose([
            RandomBackgroundNoise(16000, 'D:/musan/musan/noise')])


    def __len__(self):
        return self.N

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):

        wav_path = self.wav_list[index]

        wavS, _ = ta.load(str(wav_path), normalize=True)
        *shape, src_channels, length = wavS.shape
        if src_channels == self.channels:
            pass
        else:
            raise ValueError('Number of channels - bad')

        #example = torch.stack(wavs)
        #example = julius.resample_frac(example, meta['samplerate'], self.samplerate)
        if self.segment:
            lengthseg = int(self.segment * self.samplerate)
            hop = random.randint(0,length)
            wavS = wavS[..., hop:]
            Y = F.pad(wavS, (0, lengthseg - wavS.shape[-1]))



        return Y, self.compose_transform(Y)





